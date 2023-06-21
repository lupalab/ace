import json
import math
import os
import sys

import click
import gin
import tensorflow as tf
import tensorflow_datasets as tfds
from loguru import logger
from tensorflow.keras import mixed_precision

from ace.ace import ACEModel
from ace.masking import get_add_mask_fn, UniformMaskGenerator
from ace.utils import enable_gpu_growth, WarmUpCallback


def load_datasets(dataset, batch_size, noise_scale):
    ds = tfds.load(dataset)
    train = ds["train"].map(lambda x: x["features"]).shuffle(10000)
    val = ds["val"].map(lambda x: x["features"])
    train = train.batch(batch_size, drop_remainder=True).repeat()
    val = val.batch(batch_size)

    def add_noise(t):
        return t + tf.random.normal(tf.shape(t), stddev=noise_scale, dtype=t.dtype)

    train = train.map(add_noise)

    add_mask_fn = get_add_mask_fn(UniformMaskGenerator())
    train = train.map(add_mask_fn)
    val = val.map(add_mask_fn)

    train = train.prefetch(tf.data.AUTOTUNE)
    val = val.prefetch(tf.data.AUTOTUNE)

    num_features = train.element_spec[0].shape[-1]

    return train, val, num_features


@gin.configurable(denylist=["logdir"])
def train(
    logdir,
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

    train_data, val_data, num_features = load_datasets(dataset, batch_size, noise_scale)

    default_strategy = tf.distribute.get_strategy()
    distributed_strategy = (
        tf.distribute.MirroredStrategy() if available_gpus > 1 else default_strategy
    )

    with distributed_strategy.scope():
        model = ACEModel(num_features)

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

    class LoggingCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            logger.info(
                "[Step {}]  Energy LL: {:.3f} | Proposal LL: {:.3f} | Val Energy LL: {:.3f} | Val Proposal LL: {:.3f}",
                validation_freq * epoch,
                logs["energy_ll"],
                logs["proposal_ll"],
                logs["val_energy_ll"],
                logs["val_proposal_ll"],
            )

    logger.info("Beginning training...")

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
            tf.keras.callbacks.ModelCheckpoint(
                os.path.join(logdir, "weights.h5"),
                monitor="val_energy_ll",
                mode="max",
                save_best_only=True,
                save_weights_only=True,
            ),
            WarmUpCallback(warm_up_steps),
            LoggingCallback(),
        ],
    )

    with open(os.path.join(logdir, "history.json"), "w") as fp:
        json.dump(history.history, fp)


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
def _main(config, logdir, growth, use_mixed_precision, eager):
    os.makedirs(logdir)

    logger.remove()
    fmt = "<cyan>[{time:YYYY-MM-DD:HH:mm:ss}]</> <level>{level} -- {message}</>"
    logger.add(os.path.join(logdir, "train.log"), format=fmt)
    logger.add(sys.stderr, format=fmt, colorize=True)

    gin.parse_config_file(config)
    gin.finalize()

    tf.config.run_functions_eagerly(eager)

    if growth:
        enable_gpu_growth()

    if use_mixed_precision:
        policy = mixed_precision.Policy("mixed_float16")
        mixed_precision.set_global_policy(policy)

    train(logdir)


if __name__ == "__main__":
    _main()
