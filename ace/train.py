import os
from datetime import datetime

import click
import gin
import numpy as np
import tensorflow as tf

from ace.ace import ACE
from ace.data import load_data, get_mask_generator
from ace.utils import save_scalars, make_input_dict, enable_gpu_growth

mixed_precision = tf.keras.mixed_precision


def _disable_autosharding(*datasets):
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = (
        tf.data.experimental.AutoShardPolicy.OFF
    )
    return (d.with_options(options) for d in datasets)


@gin.configurable("train", blacklist=["_multi_gpu"])
def train_ace(
    dataset=None,
    log_dir=None,
    summary_interval=10000,
    seed=None,
    batch_size=256,
    steps=400000,
    learning_rate=5e-4,
    warm_up_steps=5000,
    use_subset_val=False,
    mask_type="bernoulli",
    num_importance_samples=20,
    load_checkpoint=None,
    energy_proposal_regularization=None,
    _multi_gpu=False,
):
    """Trains an ACE model.

    The arguments to this function should be configured in the Gin configuration file.

    Args:
        dataset: The dataset to train on. This can be a UCI dataset
            (see `ace.data.UCI_DATASETS`) or a path to directory that contains custom
            data. If using custom data, the data must be formatted in a specific way.
            See README.md for details. This argument is required.
        log_dir: The name of the directory where the model and training information
            will be saved.
        summary_interval: The frequency with which the model will be evaluated on the
            validation data and summaries will be logged to TensorBoard.
        seed: A random seed to be used by NumPy and TensorFlow.
        batch_size: The training batch size.
        steps: The number of training steps to perform.
        learning_rate: The initial learning rate.
        warm_up_steps: The number of training steps before the energy network starts
            updating.
        use_subset_val: If True, at most 2048 validation instances will be used.
        mask_type: The type of observed masks to use during training. Either
            "bernoulli" or "uniform".
        num_importance_samples: The number of importance samples to use during training
            when estimating normalizing constants.
        load_checkpoint: The path to a model checkpoint which will be used to initialize
            weights before training begins.
        energy_proposal_regularization: The coefficient of the energy-proposal penalty
            term in the loss. If None, this penalty is not used.

    Returns:
        None.
    """
    assert (
        dataset is not None
    ), 'A value for "train.dataset" must be provided in the config file.'

    # Random seeds
    if seed is not None:
        np.random.seed(seed)
        tf.random.set_seed(seed)

    available_gpus = len(tf.config.list_physical_devices("GPU"))
    if available_gpus < 2 and _multi_gpu:
        print(
            f"Warning: Multi-GPU training was enabled, but {available_gpus} GPUs are "
            f"available. Please make more GPUs available, or disable multi-GPU "
            f"training."
        )
        exit()

    default_strategy = tf.distribute.get_strategy()
    distributed_strategy = (
        tf.distribute.MirroredStrategy() if _multi_gpu else default_strategy
    )

    train_dataset, val_dataset, _ = load_data(dataset, use_subset_val)
    mask_generator = get_mask_generator(mask_type)

    train_dataset = train_dataset.shuffle(buffer_size=10000)

    train_dataset = train_dataset.batch(batch_size, drop_remainder=True).repeat()
    val_dataset = val_dataset.batch(batch_size, drop_remainder=True)

    if not isinstance(train_dataset.element_spec, tuple):

        def add_zero_missing_mask(x):
            return x, tf.zeros_like(x)

        train_dataset = train_dataset.map(add_zero_missing_mask)
        val_dataset = val_dataset.map(add_zero_missing_mask)

    def add_observed_mask(x, missing_mask):
        [observed_mask] = tf.py_function(
            mask_generator, [x.get_shape().as_list()], [tf.float32]
        )
        return x, observed_mask, missing_mask

    train_dataset = train_dataset.map(add_observed_mask)
    val_dataset = val_dataset.map(add_observed_mask)

    train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)
    val_dataset = val_dataset.prefetch(tf.data.experimental.AUTOTUNE)

    train_dataset, val_dataset = _disable_autosharding(train_dataset, val_dataset)

    train_dataset = distributed_strategy.experimental_distribute_dataset(train_dataset)
    val_dataset = distributed_strategy.experimental_distribute_dataset(val_dataset)

    if log_dir is None:
        log_dir = os.path.join("logs", dataset)
    log_dir = os.path.join(log_dir, datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
    os.makedirs(log_dir)

    with distributed_strategy.scope():
        model = ACE()

    with open(os.path.join(log_dir, "full-config.gin"), "w") as file:
        file.write(gin.operative_config_str())

    train_writer = tf.summary.create_file_writer(os.path.join(log_dir, "train"))
    val_writer = tf.summary.create_file_writer(os.path.join(log_dir, "val"))
    val_log_prob_best = -np.inf

    learning_rate_schedule = tf.keras.optimizers.schedules.PolynomialDecay(
        learning_rate, steps, end_learning_rate=1e-5
    )

    use_mixed_precision = mixed_precision.global_policy().name == "mixed_float16"

    with distributed_strategy.scope():
        optimizer = tf.optimizers.Adam(learning_rate=learning_rate_schedule)
        if use_mixed_precision:
            optimizer = mixed_precision.LossScaleOptimizer(optimizer)

    checkpoint_dir = os.path.join(log_dir, "checkpoints")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    ckpt = tf.train.Checkpoint(
        model, optimizer=optimizer, step=tf.Variable(0, dtype=tf.int64)
    )
    manager = tf.train.CheckpointManager(ckpt, checkpoint_dir, max_to_keep=1)

    def train_step(x, observed_mask, missing_mask, train_energy):
        input_dict = make_input_dict(x, observed_mask, missing_mask)

        with tf.GradientTape() as tape:
            outputs = model(
                input_dict,
                training=True,
                num_importance_samples=num_importance_samples,
            )

            energy_log_prob = tf.reduce_sum(outputs["energy_log_prob"], -1)
            proposal_log_prob = tf.reduce_sum(outputs["proposal_log_prob"], -1)

            energy_log_prob = tf.nn.compute_average_loss(
                energy_log_prob, global_batch_size=batch_size
            )
            proposal_log_prob = tf.nn.compute_average_loss(
                proposal_log_prob, global_batch_size=batch_size
            )

            loss = -(proposal_log_prob + (train_energy * energy_log_prob))

            if energy_proposal_regularization is not None:
                energy_proposal_mse = tf.nn.compute_average_loss(
                    tf.losses.mse(
                        outputs["energy_log_prob"],
                        tf.stop_gradient(outputs["proposal_log_prob"]),
                    ),
                    global_batch_size=batch_size,
                )
                loss += energy_proposal_mse * energy_proposal_mse

            if use_mixed_precision:
                loss = optimizer.get_scaled_loss(loss)

        grads = tape.gradient(loss, model.trainable_weights)
        if use_mixed_precision:
            grads = optimizer.get_unscaled_gradients(grads)

        optimizer.apply_gradients(zip(grads, model.trainable_weights))

        return energy_log_prob, proposal_log_prob

    @tf.function
    def distributed_train_step(x, observed_mask, missing_mask, train_energy):
        (
            per_replica_energy_log_prob,
            per_replica_proposal_log_prob,
        ) = distributed_strategy.run(
            train_step, args=(x, observed_mask, missing_mask, train_energy)
        )

        energy_log_prob = distributed_strategy.reduce(
            tf.distribute.ReduceOp.SUM, per_replica_energy_log_prob, axis=None
        )
        proposal_log_prob = distributed_strategy.reduce(
            tf.distribute.ReduceOp.SUM, per_replica_proposal_log_prob, axis=None
        )

        return energy_log_prob, proposal_log_prob

    energy_logp_metric = tf.keras.metrics.Mean()
    proposal_logp_metric = tf.keras.metrics.Mean()

    def eval_step(x, observed_mask, missing_mask):
        input_dict = make_input_dict(x, observed_mask, missing_mask)

        outputs = model(
            input_dict,
            training=False,
            num_importance_samples=num_importance_samples,
        )
        energy_log_prob = tf.reduce_sum(outputs["energy_log_prob"], axis=-1)
        proposal_log_prob = tf.reduce_sum(outputs["proposal_log_prob"], axis=-1)

        energy_log_prob = tf.nn.compute_average_loss(
            energy_log_prob, global_batch_size=batch_size
        )
        proposal_log_prob = tf.nn.compute_average_loss(
            proposal_log_prob, global_batch_size=batch_size
        )

        return energy_log_prob, proposal_log_prob

    @tf.function
    def distributed_eval_step(x, observed_mask, missing_mask):
        (
            per_replica_energy_log_prob,
            per_replica_proposal_log_prob,
        ) = distributed_strategy.run(eval_step, args=(x, observed_mask, missing_mask))

        energy_log_prob = distributed_strategy.reduce(
            tf.distribute.ReduceOp.SUM, per_replica_energy_log_prob, axis=None
        )
        proposal_log_prob = distributed_strategy.reduce(
            tf.distribute.ReduceOp.SUM, per_replica_proposal_log_prob, axis=None
        )

        energy_logp_metric.update_state(energy_log_prob)
        proposal_logp_metric.update_state(proposal_log_prob)

    if load_checkpoint:
        ckpt.restore(load_checkpoint).expect_partial()
        ckpt.step.assign(0)

    for x, observed_mask, missing_mask in train_dataset:
        alpha = tf.cast(ckpt.step > warm_up_steps, dtype=tf.float32)

        energy_log_prob, proposal_log_prob = distributed_train_step(
            x, observed_mask, missing_mask, alpha
        )

        if ckpt.step % summary_interval == 0 or ckpt.step == steps:
            print("Step {}".format(ckpt.step.numpy()))
            print("Energy log prob: {:.4f}".format(energy_log_prob))
            print("Proposal log prob: {:.4f}".format(proposal_log_prob))
            print("Energy Is Training: {}\n".format(alpha.numpy() == 1.0))

            values_to_log = {
                "likelihood/energy_log_prob": energy_log_prob,
                "likelihood/proposal_log_prob": proposal_log_prob,
                "misc/alpha": alpha,
                "misc/learning_rate": learning_rate_schedule(ckpt.step),
            }
            save_scalars(train_writer, ckpt.step, values_to_log)

            energy_logp_metric.reset_states()
            proposal_logp_metric.reset_states()

            for val_x, val_observed_mask, val_missing_mask in val_dataset:
                distributed_eval_step(val_x, val_observed_mask, val_missing_mask)

            save_scalars(
                val_writer,
                ckpt.step,
                {
                    "likelihood/energy_log_prob": energy_logp_metric.result(),
                    "likelihood/proposal_log_prob": proposal_logp_metric.result(),
                },
            )

            if energy_logp_metric.result() > val_log_prob_best:
                manager.save(checkpoint_number=ckpt.step)
                val_log_prob_best = energy_logp_metric.result()

        ckpt.step.assign_add(1)

        if ckpt.step > steps:
            break


@click.command(name="train")
@click.option(
    "--config",
    type=click.Path(exists=True, dir_okay=False),
    nargs=1,
    required=True,
    help="Path to the Gin configuration file to use for training.",
)
@click.option(
    "--eager",
    is_flag=True,
    help="If flag is set, eager execution will be enabled inside tf.functions.",
)
@click.option(
    "--growth/--no-growth",
    default=True,
    help="Whether or not GPU growth will be enabled.",
)
@click.option(
    "--use_mixed_precision",
    is_flag=True,
    help="If flag is set, model will be trained in mixed precision mode.",
)
@click.option(
    "--multi-gpu",
    is_flag=True,
    help="If flag is set, distributed multi-GPU training will be enabled "
    "using a mirrored distribution strategy.",
)
def _main(config, eager, growth, use_mixed_precision, multi_gpu):
    """Trains an ACE model."""
    gin.parse_config_file(config)
    tf.config.run_functions_eagerly(eager)

    if growth:
        enable_gpu_growth()

    if use_mixed_precision:
        policy = mixed_precision.Policy("mixed_float16")
        mixed_precision.set_global_policy(policy)

    train_ace(_multi_gpu=multi_gpu)


if __name__ == "__main__":
    _main()
