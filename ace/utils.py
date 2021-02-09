from typing import Dict, Union, Optional

import numpy as np
import tensorflow as tf


def save_scalars(
    writer: tf.summary.SummaryWriter, step: int, entries: Dict[str, float]
):
    """Saves a dictionary of scalars to TensorBoard.

    Args:
        writer: The `SummaryWriter` that scalars should be saved to.
        step: The time step associated with the saved values.
        entries: A dictionary where each entry has a string key and a scalar value.

    Returns:
        None.
    """
    with writer.as_default():
        for k, v in entries.items():
            tf.summary.scalar(k, v, step)


def nrmse_score(
    imputations: np.ndarray, true_data: np.ndarray, observed_mask: np.ndarray
) -> np.ndarray:
    """Calculates the normalized root-mean-square error metric for imputed values.

    Args:
        imputations: A numpy array of shape `[..., num_instances, num_features]` that
            contains the imputed data.
        true_data: A numpy array of shape `[..., num_instances, num_features]` that
            contains the ground truth data.
        observed_mask: A numpy array of shape `[..., num_instances, num_features]` that
            contains the binary mask indicating which features are observed.

    Returns:
        A numpy array with shape `imputations.shape[:-2]` that contains the
        NRMSE scores.
    """
    error = (imputations - true_data) ** 2
    mse = np.sum(error, axis=-2) / np.count_nonzero(1.0 - observed_mask, axis=-2)
    nrmse = np.sqrt(mse) / np.std(true_data, axis=-2)
    return np.mean(nrmse, axis=-1)


def make_input_dict(
    x: Union[np.ndarray, tf.Tensor],
    observed_mask: Union[np.ndarray, tf.Tensor],
    missing_mask: Optional[Union[np.ndarray, tf.Tensor]] = None,
) -> Dict[str, Union[np.ndarray, tf.Tensor]]:
    """Packages ACE inputs into the appropriate dictionary representation.

    Args:
        x: The data values being fed into ACE.
        observed_mask: The observed mask being fed into ACE.
        missing_mask: The optional missing mask being fed into ACE.

    Returns:
        A dictionary that can be passed into an `ACE` model when it is called.
    """
    input_dict = {"x": x, "observed_mask": observed_mask}
    if missing_mask is not None:
        input_dict["missing_mask"] = missing_mask
    return input_dict


def enable_gpu_growth():
    """Enables GPU growth for TensorFlow."""
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            for dev in gpus:
                tf.config.experimental.set_memory_growth(dev, True)
        except RuntimeError as e:
            print(e)
