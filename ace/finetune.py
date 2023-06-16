import json
import math
import os
import sys

import click
import gin
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_probability as tfp
from loguru import logger
from tensorflow.keras import mixed_precision
import numpy as np
from tqdm import tqdm

import pdb

from ace.ace_proposal import ACEModel
from ace.masking import get_add_mask_fn, UniformMaskGenerator, FixedMaskGenerator, SubsetMaskGenerator
from ace.utils import enable_gpu_growth, WarmUpCallback, load_model
from ace.evaluation import evaluate_imputation, nrmse_score

class FinetunePerformanceTracker:
    def __init__(self, eval_function):
        self.eval_function = eval_function
        self.performance_history = []
    
    def evaluate(self, *args):
        result = self.eval_function(*args)
        self.performance_history.append(result)
        return result


def load_datasets(dataset, batch_size, noise_scale, mask_fn=UniformMaskGenerator):
    ds = tfds.load(dataset)
    train = ds["train"].map(lambda x: x["features"]).shuffle(10000)
    val = ds["val"].map(lambda x: x["features"])
    train = train.batch(batch_size, drop_remainder=True).repeat()
    val = val.batch(batch_size)

    def add_noise(t):
        return t + tf.random.normal(tf.shape(t), stddev=noise_scale, dtype=t.dtype)

    train = train.map(add_noise)

    add_mask_fn = get_add_mask_fn(mask_fn)
    train = train.map(add_mask_fn)
    val = val.map(add_mask_fn)

    train = train.prefetch(tf.data.AUTOTUNE)
    val = val.prefetch(tf.data.AUTOTUNE)

    num_features = train.element_spec[0].shape[-1]

    return train, val, num_features

@gin.configurable(denylist=["logdir"])
def finetune(
    logdir,
    model_dir,
    masks,
    dataset=gin.REQUIRED,
    batch_size=10000,
    noise_scale=0.001,
    learning_rate=5e-4,
    steps=1000000,
    warm_up_steps=1000,
    validation_freq=5000,
    validation_steps=None,
    regpen=0.01,
    nsubmasks=None,
    maskbern=0.5,
    mask_batch_size=64,
    hidden_size=512,
    seed=None
):
    available_gpus = len(tf.config.list_physical_devices("GPU"))
    if available_gpus > 0:
        logger.info("Using {} found GPU(s).", available_gpus)


    stateless_seed = tfp.random.sanitize_seed(seed)

    current_mask = masks[0, None]
    fixed_mask_generator = FixedMaskGenerator(current_mask)
    subset_mask_generator = SubsetMaskGenerator(current_mask)
    fixed_mask_fn = lambda shape: fixed_mask_generator(shape)
    subset_mask_fn = lambda shape: subset_mask_generator(shape)

    default_strategy = tf.distribute.get_strategy()
    distributed_strategy = (
        tf.distribute.MirroredStrategy() if available_gpus > 1 else default_strategy
    )

    with distributed_strategy.scope():
        with open(os.path.join(model_dir, "model_config.json"), "r") as fp:
            model_config = json.load(fp)

        model_config["regpen"] = regpen
        
        model = ACEModel.from_config(model_config)
        model.load_weights(os.path.join(model_dir, "weights.h5"))

        feature_network = tf.keras.Model(model._proposal_network.input, model._proposal_network.layers[-6].output)
        feature_network.trainable = False

        def create_proposal_dist(t):

            tf.random.set_seed(seed)

            logits = t[..., :k]
            means = t[..., k:-k]
            scales = tf.nn.softplus(t[..., -k:]) + 1e-3
            components_dist = tfp.distributions.Normal(
                loc=tf.cast(means, tf.float32), scale=tf.cast(scales, tf.float32)
            )
            return tfp.distributions.MixtureSameFamily(
                mixture_distribution=tfp.distributions.Categorical(
                    logits=tf.cast(logits, tf.float32)
                ),
                components_distribution=components_dist
            )

        def safe_sample(dist):
            return dist.sample(seed=stateless_seed)

        proposal_layer = tfp.layers.DistributionLambda(create_proposal_dist, convert_to_tensor_fn=safe_sample)
        proposal_layer.trainable = False

    config = model.get_config()

    with open(os.path.join(logdir, "model_config.json"), "w") as fp:
        json.dump(config, fp)

    with open(os.path.join(logdir, "operative-config.gin"), "w") as fp:
        fp.write(gin.operative_config_str())

    logger.info("Constructed ACE model with {} parameters.", model.count_params())

    baseline_weights = model.finetune_layer.get_weights()

    baseline_kernel = baseline_weights[0]
    baseline_bias = baseline_weights[1]

    model.finetune_kernel_reg.set_baseline(baseline_kernel)
    model.finetune_bias_reg.set_baseline(baseline_bias)

    stop = False

    def impute_metric():
        (   energy_nrmse,
            proposal_nrmse,
            energy_imputations,
            proposal_imputations,
        ) = evaluate_imputation(
            model,
            tfds.load(dataset)["test"].map(lambda x: x["features"]).batch(batch_size),
            fixed_mask_generator,
            mask_fn=fixed_mask_fn,
            num_trials=1,
            num_importance_samples=20000,
        )

        return np.mean(proposal_nrmse)

    class LoggingCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            logger.info(
                "[Step {}] Proposal LL: {:.3f} | Val Proposal LL: {:.3f}",
                validation_freq * epoch,
                logs["proposal_ll"],
                logs["val_proposal_ll"],
            )
            logger.info(
                f"Imputation NRMSE: {tracker.evaluate()}"
            )

    mask_histories = []

    @tf.function
    def finetune_step(model, batch):
        
        x, b = batch

        x_o, x_u, observed_mask, query = model._process_inputs(x, b, missing_mask=None)

        feats = feature_network([x_o, observed_mask])
    
        with tf.GradientTape() as tape:
            output = model.finetune_layer(feats)
            output = tf.reshape(output, [-1, config["num_features"], 3 * k + context_units])

            params = output[..., context_units:]

            proposal_dist = proposal_layer(params)

            proposal_ll = proposal_dist.log_prob(tf.cast(x_u, tf.float32))
            proposal_ll *= tf.cast(query, tf.float32)

            # proposal_samples = tf.stop_gradient(
            #     proposal_dist.sample(10)
            # )
            # proposal_samples = tf.transpose(proposal_samples, [1, 0, 2])
            # proposal_samples *= tf.expand_dims(tf.cast(query, tf.float32), 1)

            proposal_mean = proposal_dist.mean() * tf.cast(query, tf.float32)

            imputations = x_o + proposal_mean

            error = (imputations - x) ** 2
            mse = tf.reduce_sum(error, axis=-2) / tf.cast(tf.maximum(tf.math.count_nonzero(1.0 - observed_mask, axis=-2), 1), tf.float32)
            nrmse = tf.math.sqrt(mse) / tf.math.reduce_std(x, axis=-2)
            # nrmse = nrmse_score(imputations, x, observed_mask)

            loss = tf.reduce_mean(tf.boolean_mask(nrmse, tf.math.greater(nrmse, 0)))

        linear_grads = tape.gradient(loss, model.finetune_layer.trainable_weights)
        optimizer.apply_gradients(zip(linear_grads, model.finetune_layer.trainable_weights))

        return loss  

    lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(
        learning_rate, steps, end_learning_rate=1e-7
    )
    optimizer = tf.keras.optimizers.Adam(lr_schedule)

    for m in range(len(masks)):            

        current_mask = masks[m, None]

        print(current_mask)

        if tf.reduce_sum(current_mask) < 1:
            continue

        fixed_mask_generator = FixedMaskGenerator(current_mask)
        subset_mask_generator = SubsetMaskGenerator(current_mask)

        # impute_features = lambda: tf.reshape(tf.where(1-tf.squeeze(current_mask)), (-1))

        train_data, val_data, num_features = load_datasets(dataset, batch_size, noise_scale, mask_fn=fixed_mask_fn)

        train_X, train_B, *train_M = next(iter(train_data))
        val_X, val_B, *val_M = next(iter(val_data))
        
        tracker = FinetunePerformanceTracker(impute_metric)

        # logger.info(f"Pre-trained NMRSE: {tracker.evaluate()}")


        with distributed_strategy.scope():
            model.finetune_layer.set_weights(baseline_weights)

            for var in optimizer.variables():
                var.assign(tf.zeros_like(var))


        # print(optimizer._current_learning_rate)

        # model.compile(
        #     optimizer,
        #     # loss = lambda true, pred: main_loss(true, pred) + 
        #     )
        # beta_init = model.finetune_layer.

        # main_loss = tf.keras.losses.SparseCategoricalCrossEntropy()

        logger.info("Beginning finetuning...")

        k = config["mixture_components"]
        context_units = config["context_units"]

        loss_hist = []

        for i, batch in enumerate(train_data):

            if i % validation_freq == 0:
                logger.info(f"Finetuned on {i} batches; Imputation NRMSE = {tracker.evaluate()}")

            if i >= steps:
                break

            loss = distributed_strategy.run(finetune_step, args=(model, batch))
            loss_hist.append(loss)

            if np.any(np.isnan(distributed_strategy.experimental_local_results(loss)[0].numpy())):
                pdb.set_trace()           
            

        # logger.info(
        #             f"Imputation NRMSE: {tracker.evaluate()}"
        #         )

        # datafeats = model.feats([train_X, train_B], *train_M)

        # if nsubmasks is not None:

        #     submasks = tf.cast(
        #         tf.random.uniform([nsubmasks, 1, train_data.shape[1]]) <= maskbern,
        #         tf.float32) * current_mask[None, :, :]

        #     submasks = tf.boolean_mask(submasks, tf.reduce_sum(
        #         submasks, axis=[1, 2]) > 0, axis=0)
        #     nsubmasks = submasks.shape[0]

        #     regbeta = tf.concat(
        #         [0.5*tf.expand_dims(tf.transpose(model.finetune_layer), -1), 0.5*tf.ones((train_data.shape[1], nsubmasks, 1))/nsubmasks], 1)

        #     datafeats = tf.concat(
        #         [tf.stack([datafeats] * train_X.shape[1]), tf.transpose(tf.concat([model(tf.expand_dims(train_X, axis=0), submasks[i:i+mask_batch_size]) 
        #                                                                             for i in range(0, nsubmasks, mask_batch_size)], axis=0), perm=(2, 1, 0))], -1)

        #     xfeats = model.feats([val_X, val_B], *val_M)

        #     databeta = tf.concat([tf.linalg.solve(tf.matmul(datafeats[i:i+1], datafeats[i:i+1], transpose_a=True) + regpen*tf.eye(hidden_size+nsubmasks),
        #                         tf.matmul(datafeats[i:i+1], train_data[:, i:i+1], transpose_a=True)+regpen*regbeta[i]) for i in impute_features()], 0)

        #     xfeats = tf.concat(
        #         [tf.stack([xfeats] * impute_features.shape[0]), 
        #         tf.gather(tf.transpose(tf.concat([model(tf.expand_dims(val_data, axis=0), submasks[i:i+mask_batch_size]) 
        #                                         for i in range(0, nsubmasks, mask_batch_size)], axis=0), 
        #                                         perm=(2, 1, 0)), impute_features(), axis=0)], -1)
        
        # else:
        #     regbeta = tf.expand_dims(tf.transpose(model.finetune_layer), -1)

        #     databeta = tf.concat([tf.linalg.solve(
        #         tf.matmul(datafeats[i], datafeats[i],
        #                     transpose_a=True)+regpen*tf.eye(hidden_size),
        #         tf.matmul(datafeats[i], train_X, transpose_a=True)+regpen*regbeta[i]) for i in range(train_X.shape[1])], 1)

        # history = model.fit(
        #     train_data,
        #     validation_data=val_data,
        #     validation_steps=validation_steps,
        #     epochs=int(math.ceil(steps / validation_freq)),
        #     steps_per_epoch=validation_freq,
        #     verbose=0,
        #     callbacks=[
        #         tf.keras.callbacks.TensorBoard(
        #             logdir,
        #             update_freq=validation_freq,
        #             write_graph=False,
        #             profile_batch=(5, 10),
        #         ),
        #         # tf.keras.callbacks.ModelCheckpoint(
        #         #     os.path.join(logdir, "weights.h5"),
        #         #     monitor="val_proposal_ll",
        #         #     mode="max",
        #         #     save_best_only=True,
        #         #     save_weights_only=True,
        #         # ),
        #         WarmUpCallback(warm_up_steps),
        #         LoggingCallback(),
        #     ],
        # )

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

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    gpus = tf.config.list_physical_devices('GPU')
    tf.config.set_visible_devices(gpus[0:2], "GPU")

    tf.config.run_functions_eagerly(eager)
    seed = 123
    print(tfp.random.sanitize_seed(seed))
    tf.random.set_seed(seed)

    if growth:
        enable_gpu_growth()

    if use_mixed_precision:
        policy = mixed_precision.Policy("mixed_float16")
        mixed_precision.set_global_policy(policy)

    results = []

    masks = tf.cast(tf.random.uniform([100, 8]) <= 0.5, tf.float32)

    results = finetune(logdir, model_dir, masks, steps=20000, learning_rate=5e-4, validation_freq=5000, seed=seed)

    results = list(zip(*results))
    logger.info(str(results))

    np.save(os.path.join(logdir, "impute_results.npy"), np.array(results))

if __name__ == "__main__":
    _main()
