import json
import math
import os
import sys
import numpy as np

import click
import gin
import tensorflow as tf
import tensorflow_datasets as tfds
from loguru import logger
from tensorflow.keras import mixed_precision

from ace.ace_proposal import ACEModel
from ace.masking import get_add_mask_fn, FixedMaskGenerator, BernoulliMaskGenerator, enumerate_mask
from ace.utils import enable_gpu_growth, WarmUpCallback, get_config_dict
from ace.evaluation import evaluate_imputation

class FinetunePerformanceTracker:
    def __init__(self, eval_function):
        self.eval_function = eval_function
        self.performance_history = []

    def evaluate(self, *args):
        result = self.eval_function(*args)
        self.performance_history.append(result)
        return result


def load_datasets(dataset, batch_size, noise_scale, mask_fn):
    ds = tfds.load(dataset)
    train = ds["train"].map(lambda x: x["features"]).shuffle(10000)
    val = ds["val"].map(lambda x: x["features"])
    test = ds["test"].map(lambda x: x["features"])
    train = train.batch(batch_size, drop_remainder=True).repeat()
    val = val.batch(batch_size)
    test = test.batch(batch_size)

    def add_noise(t):
        return t + tf.random.normal(tf.shape(t), stddev=noise_scale, dtype=t.dtype)

    train = train.map(add_noise)

    add_mask_fn = get_add_mask_fn(mask_fn)
    train = train.map(add_mask_fn)
    val = val.map(add_mask_fn)
    test = test.map(add_mask_fn)

    train = train.prefetch(tf.data.AUTOTUNE)
    val = val.prefetch(tf.data.AUTOTUNE)
    test = test.prefetch(tf.data.AUTOTUNE)

    num_features = train.element_spec[0].shape[-1]

    return train, val, test, num_features


@gin.configurable(denylist=["logdir"])
def finetune(
    logdir,
    model_dir,
    masks,
    dataset=gin.REQUIRED,
    batch_size=512,
    noise_scale=0.001,
    learning_rate=5e-4,
    steps=1000000,
    warm_up_steps=10000,
    validation_freq=5000,
    validation_steps=None,
):
    available_gpus = len(tf.config.list_physical_devices("GPU"))
    if available_gpus > 0:
        logger.info("Using {} found GPU(s).", available_gpus)

    current_mask = masks[0]
    fixed_mask_generator = FixedMaskGenerator(current_mask)
    fixed_mask_fn = lambda shape: fixed_mask_generator(shape)
    bern_mask_generator = BernoulliMaskGenerator()

    dataset = get_config_dict(model_dir)["train"]["dataset"]
    train_data, val_data, test_data, num_features = load_datasets(dataset, batch_size, noise_scale, mask_fn=fixed_mask_generator)

    default_strategy = tf.distribute.get_strategy()
    distributed_strategy = (
        tf.distribute.MirroredStrategy() if available_gpus > 1 else default_strategy
    )

    with distributed_strategy.scope():
        model = ACEModel(num_features)
    
    model.load_weights(os.path.join(model_dir, "weights.h5"))
    # model._proposal_network.trainable = False
    # model.finetune_layer.trainable = True
    baseline_weights = model.finetune_layer.get_weights()

    baseline_kernel = baseline_weights[0]
    baseline_bias = baseline_weights[1]

    model.finetune_kernel_reg.set_baseline(baseline_kernel)
    model.finetune_bias_reg.set_baseline(baseline_bias)

    logger.info("Constructed ACE model with {} parameters.", model.count_params())

    with open(os.path.join(logdir, "model_config.json"), "w") as fp:
        json.dump(model.get_config(), fp)

    with open(os.path.join(logdir, "operative-config.gin"), "w") as fp:
        fp.write(gin.operative_config_str())

    lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(
        learning_rate, steps, end_learning_rate=1e-7
    )
    optimizer = tf.keras.optimizers.Adam(lr_schedule)

    model.compile(optimizer)

    def impute_metric():
        (   energy_nrmse,
            proposal_nrmse,
            energy_imputations,
            proposal_imputations,
        ) = evaluate_imputation(
            model,
            test_data,
            fixed_mask_generator,
            mask_fn=fixed_mask_fn,
            num_trials=1,
            num_importance_samples=20000,
        )

        print(proposal_nrmse)

        return np.mean(proposal_nrmse)


    class LoggingCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            logger.info(
                "[Step {}] Proposal LL: {:.3f} | Val Proposal LL: {:.3f}",
                validation_freq * epoch,
                # logs["energy_ll"],
                logs["proposal_ll"],
                # logs["val_energy_ll"],
                logs["val_proposal_ll"],
            )
            logger.info(
                f"Imputation NRMSE: {tracker.evaluate()}"
            )

    logger.info("Beginning training...")

    mask_histories = []

    for m in range(len(masks)):

        current_mask = masks[m]

        print(current_mask)

        if np.prod(current_mask) > 0:
            continue

        fixed_mask_generator = FixedMaskGenerator(current_mask)

        # impute_features = lambda: tf.reshape(tf.where(1-tf.squeeze(current_mask)), (-1))

        train_data, val_data, test_data, num_features = load_datasets(dataset, batch_size, noise_scale, mask_fn=fixed_mask_generator)

        train_X, train_B, *train_M = next(iter(train_data))
        val_X, val_B, *val_M = next(iter(val_data))

        tracker = FinetunePerformanceTracker(impute_metric)

        with distributed_strategy.scope():
            # model.finetune_layer.set_weights(baseline_weights)
            model.load_weights(os.path.join(model_dir, "weights.h5"))

            for var in optimizer.variables():
                var.assign(tf.zeros_like(var))
        
        logger.info(
                f"Pretrained NRMSE: {tracker.evaluate()}"
            )

        logger.info("Beginning finetuning...")

        history = model.fit(
            train_data,
            validation_data=val_data,
            validation_steps=validation_steps,
            epochs=int(math.ceil(steps / validation_freq)),
            steps_per_epoch=validation_freq,
            verbose=0,
            callbacks=[
                tf.keras.callbacks.TensorBoard(
                    logdir,
                    update_freq=validation_freq,
                    write_graph=False,
                    profile_batch=(5, 10),
                ),
                # tf.keras.callbacks.ModelCheckpoint(
                #     os.path.join(logdir, "weights.h5"),
                #     monitor="val_energy_ll",
                #     mode="max",
                #     save_best_only=True,
                #     save_weights_only=True,
                # ),
                WarmUpCallback(warm_up_steps),
                LoggingCallback(),
            ],
        )

        mask_histories.append(tracker.performance_history)

    # with open(os.path.join(logdir, "history.json"), "w") as fp:
    #     json.dump(history.history, fp)

    return mask_histories


@click.command("train")
@click.option(
    "--config",
    type=click.Path(exists=True, dir_okay=False),
    nargs=1,
    required=True,
    help="Path to the Gin configuration file to use for training.",
)
@click.option(
    "--logdir",
    type=click.Path(dir_okay=True, file_okay=False),
    nargs=1,
    required=True,
    help="Path to the directory where the experiment will be logged.",
)
@click.option(
    "--model_dir",
    type=click.Path(exists=True, file_okay=False),
    nargs=1,
    required=True,
    help="Path of the directory of the model to evaluate.",
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
@click.option("--eager", is_flag=True, help="Whether or not to use eager execution.")
def _main(config, logdir, model_dir, growth, use_mixed_precision, eager):
    os.makedirs(logdir, exist_ok=True)

    logger.remove()
    fmt = "<cyan>[{time:YYYY-MM-DD:HH:mm:ss}]</> <level>{level} -- {message}</>"
    logger.add(os.path.join(logdir, "finetune.log"), format=fmt)
    logger.add(sys.stderr, format=fmt, colorize=True)

    gin.parse_config_file(config)
    gin.finalize()

    tf.config.run_functions_eagerly(eager)

    # seed = 42
    # tf.random.set_seed(seed)

    # if growth:
    #     enable_gpu_growth()

    if use_mixed_precision:
        policy = mixed_precision.Policy("mixed_float16")
        mixed_precision.set_global_policy(policy)

    results = []

    d = 8

    masks = np.stack([enumerate_mask(d, i) for i in range(2**d)])

    results = finetune(logdir, model_dir, masks, steps=0, learning_rate=5e-5, validation_freq=5000, batch_size=32)

    results = list(zip(*results))
    logger.info(str(results))

    np.save(os.path.join(logdir, "impute_results.npy"), np.array(results))


if __name__ == "__main__":
    _main()
