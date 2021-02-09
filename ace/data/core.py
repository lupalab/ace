import glob
import os
from typing import Tuple, Union

import numpy as np
import pandas as pd
import tensorflow as tf

from ace.data.masking import BernoulliMaskGenerator

UCI_DATASETS = {
    "gas",
    "gas-missing-10",
    "gas-missing-50",
    "power",
    "power-missing-10",
    "power-missing-50",
    "hepmass",
    "hepmass-missing-10",
    "hepmass-missing-50",
    "miniboone",
    "miniboone-missing-10",
    "miniboone-missing-50",
    "bsds",
    "bsds-missing-10",
    "bsds-missing-50",
}


def get_dataset_type(dataset: str) -> Union[str, None]:
    """Gets the type of a dataset.

    Args:
        dataset: The requested dataset.

    Returns:
        The type of `dataset` (e.g. "UCI"), or None if an invalid dataset was
        provided.
    """
    if dataset in UCI_DATASETS:
        return "UCI"

    if os.path.isdir(dataset):
        return "file"


def load_data(
    dataset: str,
    use_subset_val: bool = False,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """Loads the requested dataset as a `tf.data.Dataset`.

    Args:
        dataset: The dataset to load.
        use_subset_val: If True, only a small subset of the validation data will
            be returned in the dataset.

    Returns:
        A tuple with three `tf.data.Dataset`s containing the train, validation,
        and test partitions.
    """
    dataset_type = get_dataset_type(dataset)

    if dataset_type == "UCI":
        train_dataset, val_dataset, test_dataset = load_uci_dataset(
            dataset, use_subset_val
        )
    elif dataset_type == "file":
        train_dataset, val_dataset, test_dataset = load_file_dataset(
            dataset, use_subset_val
        )
    else:
        raise ValueError(f"'{dataset}' is neither a UCI dataset nor a valid directory.")

    return train_dataset, val_dataset, test_dataset


def load_uci_dataset(
    dataset: str,
    use_subset_val: bool = False,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """Loads a UCI dataset as a `tf.data.Dataset`.

    When the requested dataset specifies missing data, each element in the returned
    dataset is a tuple `(x, missing_mask)`, where `missing_mask` is a binary
    mask with the same shape as `x` that indicates which values of `x` should be
    considered missing. The same random seed is always used to generate these
    masks.

    Args:
        dataset: The UCI dataset to load.
        use_subset_val: If True, only a small subset of the validation data will
            be returned in the dataset.

    Returns:
        A tuple with three `tf.data.Dataset`s containing the train, validation,
        and test partitions.
    """
    missing_rate = None
    if "missing" in dataset:
        dataset, missing_rate = dataset.split("-missing-")
        missing_rate = float(missing_rate) / 100

    data_train, data_val, data_test = load_uci_np(dataset)

    if missing_rate is not None:
        mask_generator = BernoulliMaskGenerator(p=missing_rate, seed=91)

        def add_mask(t):
            [mask] = tf.py_function(
                mask_generator, [t.get_shape().as_list()], [tf.float32]
            )
            return t, mask

        train_dataset = tf.data.Dataset.from_tensor_slices(data_train)
        train_dataset = train_dataset.map(add_mask).cache()

        val_dataset = tf.data.Dataset.from_tensor_slices(data_val)
        if use_subset_val:
            val_dataset = val_dataset.take(2048)
        val_dataset = val_dataset.map(add_mask).cache()

        test_dataset = tf.data.Dataset.from_tensor_slices(data_test)
        test_dataset = test_dataset.map(add_mask).cache()
    else:
        train_dataset = tf.data.Dataset.from_tensor_slices(data_train)
        train_dataset = train_dataset.shuffle(buffer_size=10000)

        val_dataset = tf.data.Dataset.from_tensor_slices(data_val)
        if use_subset_val:
            val_dataset = val_dataset.take(2048)

        test_dataset = tf.data.Dataset.from_tensor_slices(data_test)

    return train_dataset, val_dataset, test_dataset


def load_uci_np(
    dataset: str, data_root: str = "data"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Loads a UCI dataset as numpy arrays.

    Args:
        dataset: The dataset to load.
        data_root: The directory where the data is stored.

    Returns:
        The tuple (train_data, val_data, test_data).
    """
    data_path = os.path.join(data_root, "processed", dataset)
    if not os.path.exists(data_path):
        print(f"Could not find the requested data at: {data_path}")
        print("In order to use UCI/BSDS data, run the following command:\n")
        print("     python -m ace.data.download\n")
        exit()

    train_data = np.load(os.path.join(data_path, "train.npy"))
    val_data = np.load(os.path.join(data_path, "val.npy"))
    test_data = np.load(os.path.join(data_path, "test.npy"))

    return (
        train_data.astype(np.float32),
        val_data.astype(np.float32),
        test_data.astype(np.float32),
    )


def load_file_dataset(
    dataset_path: str,
    use_subset_val: bool = False,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """Loads a user's dataset from file, as a `tf.data.Dataset`.

    This method expects that the provided directory contains exactly three files
    in one of two formats:
        train.csv, val.csv, test.csv
            or
        train.npy, val.npy, test.npy

    These files will contain the training data, validation data, and test data. It is
    okay if the data contains missing values. In CSV format, the missing values should
    be interpreted as NaN by Pandas. In NumPy format, the missing values should be NaN.

    Args:
        dataset_path: The path to the dataset to load.
        use_subset_val: If True, only a small subset of the validation data will
            be returned in the dataset.

    Returns:
        A tuple with three `tf.data.Dataset`s containing the train, validation,
        and test partitions.
    """
    dir_files = glob.glob(os.path.join(dataset_path, "*"))

    if len(dir_files) != 3:
        raise ValueError(
            f"The data directory should have 3 files (one for train, val, and test). "
            f"However, the provided directory has {len(dir_files)}."
        )

    file_type = os.path.splitext(dir_files[0])[1][1:]

    if file_type == "csv":
        train, train_missing, val, val_missing, test, test_missing = load_csv(
            dataset_path
        )
    elif file_type == "npy":
        train, train_missing, val, val_missing, test, test_missing = load_npy(
            dataset_path
        )
    else:
        raise ValueError(
            f"'.{file_type}' is not a valid data format. "
            f"Must be either '.csv' or '.npy'."
        )

    train_dataset = tf.data.Dataset.zip(
        (
            tf.data.Dataset.from_tensor_slices(train),
            tf.data.Dataset.from_tensor_slices(train_missing),
        )
    )
    train_dataset = train_dataset.shuffle(buffer_size=10000)
    train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

    val_dataset = tf.data.Dataset.zip(
        (
            tf.data.Dataset.from_tensor_slices(val),
            tf.data.Dataset.from_tensor_slices(val_missing),
        )
    )
    if use_subset_val:
        val_dataset = val_dataset.take(2048)
    val_dataset = val_dataset.prefetch(tf.data.experimental.AUTOTUNE)

    test_dataset = tf.data.Dataset.zip(
        (
            tf.data.Dataset.from_tensor_slices(test),
            tf.data.Dataset.from_tensor_slices(test_missing),
        )
    )

    return train_dataset, val_dataset, test_dataset


def load_csv(data_dir):
    train = pd.read_csv(os.path.join(data_dir, "train.csv"), header=0)
    val = pd.read_csv(os.path.join(data_dir, "val.csv"), header=0)
    test = pd.read_csv(os.path.join(data_dir, "test.csv"), header=0)

    train_missing = train.isna().to_numpy(dtype=np.float32)
    val_missing = val.isna().to_numpy(dtype=np.float32)
    test_missing = test.isna().to_numpy(dtype=np.float32)

    train = train.fillna(0).to_numpy(dtype=np.float32)
    val = val.fillna(0).to_numpy(dtype=np.float32)
    test = test.fillna(0).to_numpy(dtype=np.float32)

    return train, train_missing, val, val_missing, test, test_missing


def load_npy(data_dir):
    train = np.load(os.path.join(data_dir, "train.npy")).astype(np.float32)
    val = np.load(os.path.join(data_dir, "train.npy")).astype(np.float32)
    test = np.load(os.path.join(data_dir, "train.npy")).astype(np.float32)

    train_missing = np.isnan(train).astype(np.float32)
    val_missing = np.isnan(val).astype(np.float32)
    test_missing = np.isnan(test).astype(np.float32)

    train = np.nan_to_num(train, copy=False)
    val = np.nan_to_num(val, copy=False)
    test = np.nan_to_num(test, copy=False)

    return train, train_missing, val, val_missing, test, test_missing
