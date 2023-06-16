import json
import os
from typing import Dict, Any

import tensorflow as tf

from gin.config_parser import ConfigParser

from ace import ACEModel

def enable_gpu_growth():
    """Enables GPU growth for TensorFlow."""
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            for dev in gpus:
                tf.config.experimental.set_memory_growth(dev, True)
        except RuntimeError as e:
            print(e)


def load_model(model_dir: str, with_saved_weights: bool = True) -> ACEModel:
    """Loads an ACEModel from its training directory.

    Args:
        model_dir: The directory for the model to load.
        with_saved_weights: If True, the model's saved weights will be loaded.

    Returns:
        An ACEModel.
    """
    with open(os.path.join(model_dir, "model_config.json"), "r") as fp:
        model_config = json.load(fp)

    model = ACEModel.from_config(model_config)

    if with_saved_weights:
        model.load_weights(os.path.join(model_dir, "weights.h5"))

    return model


def get_config_dict(model_dir: str) -> Dict[str, Any]:
    class DummyConfigurableReferenceType:
        def __init__(self, scoped_configurable_name, evaluate):
            pass

    with open(os.path.join(model_dir, "operative-config.gin"), "r") as fp:
        parser = ConfigParser(fp, DummyConfigurableReferenceType)
        config = {}
        for bind in parser:
            config.setdefault(bind.selector, {})[bind.arg_name] = bind.value

    return config


class WarmUpCallback(tf.keras.callbacks.Callback):
    def __init__(self, warm_up_steps):
        super().__init__()
        self._warm_up_steps = warm_up_steps
        self._total_batches = 0

    def on_train_begin(self, logs=None):
        self.model._alpha.assign(0.0)

    def on_train_batch_end(self, batch, logs=None):
        if self._total_batches == self._warm_up_steps:
            self.model._alpha.assign(1.0)

        self._total_batches += 1



