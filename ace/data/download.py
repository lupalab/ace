"""Utilities for downloading and preprocessing the benchmark UCI datasets.

This module was adapted from:
https://github.com/conormdurkan/autoregressive-energy-machines/blob/master/tensorflow/utils/data_utils.py
"""

import os
import pathlib
import shutil
import tarfile
from collections import Counter
from typing import Tuple
from urllib import request

import click
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm


def preprocess_gas(
    data_root: str = "data",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Preprocesses the Gas dataset.

    Args:
        data_root: The directory where the data is stored.

    Returns:
        A tuple with the train, validation, and test partitions of the data.
    """

    def load_data(file):
        data = pd.read_pickle(file)
        data.drop("Meth", axis=1, inplace=True)
        data.drop("Eth", axis=1, inplace=True)
        data.drop("Time", axis=1, inplace=True)
        return data

    def get_correlation_numbers(data):
        C = data.corr()
        A = C > 0.98
        B = A.sum(axis=1)
        return B

    def load_data_and_clean(file):
        data = load_data(file)
        B = get_correlation_numbers(data)

        while np.any(B > 1):
            col_to_remove = np.where(B > 1)[0][0]
            col_name = data.columns[col_to_remove]
            data.drop(col_name, axis=1, inplace=True)
            B = get_correlation_numbers(data)
        data = (data - data.mean()) / data.std()

        return data.values

    def load_data_and_clean_and_split(file):
        data = load_data_and_clean(file)
        N_test = int(0.1 * data.shape[0])
        data_test = data[-N_test:]
        data_train = data[0:-N_test]
        N_validate = int(0.1 * data_train.shape[0])
        data_validate = data_train[-N_validate:]
        data_train = data_train[0:-N_validate]

        return data_train, data_validate, data_test

    file = os.path.join(data_root, "raw/gas/ethylene_CO.pickle")
    return load_data_and_clean_and_split(file)


def preprocess_power(
    data_root: str = "data",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Preprocesses the Power dataset.

    Args:
        data_root: The directory where the data is stored.

    Returns:
        A tuple with the train, validation, and test partitions of the data.
    """

    def load_data_split_with_noise(data):
        rng = np.random.RandomState(42)
        rng.shuffle(data)
        N = data.shape[0]

        data = np.delete(data, 3, axis=1)
        data = np.delete(data, 1, axis=1)
        ############################
        # Add noise
        ############################
        voltage_noise = 0.01 * rng.rand(N, 1)
        gap_noise = 0.001 * rng.rand(N, 1)
        sm_noise = rng.rand(N, 3)
        time_noise = np.zeros((N, 1))
        noise = np.hstack((gap_noise, voltage_noise, sm_noise, time_noise))
        data = data + noise

        N_test = int(0.1 * data.shape[0])
        data_test = data[-N_test:]
        data = data[0:-N_test]
        N_validate = int(0.1 * data.shape[0])
        data_validate = data[-N_validate:]
        data_train = data[0:-N_validate]

        return data_train, data_validate, data_test

    def load_data_normalised(data):
        data_train, data_validate, data_test = load_data_split_with_noise(data)
        data = np.vstack((data_train, data_validate))
        mu = data.mean(axis=0)
        s = data.std(axis=0)
        data_train = (data_train - mu) / s
        data_validate = (data_validate - mu) / s
        data_test = (data_test - mu) / s

        return data_train, data_validate, data_test

    file = os.path.join(data_root, "raw/power/data.npy")
    data = np.load(file)
    return load_data_normalised(data)


def preprocess_hepmass(
    data_root: str = "data",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Preprocesses the Hepmass dataset.

    Args:
        data_root: The directory where the data is stored.

    Returns:
        A tuple with the train, validation, and test partitions of the data.
    """

    def load_data(path):
        data_train = pd.read_csv(
            filepath_or_buffer=os.path.join(path, "1000_train.csv"), index_col=False
        )
        data_test = pd.read_csv(
            filepath_or_buffer=os.path.join(path, "1000_test.csv"), index_col=False
        )
        return data_train, data_test

    def load_data_no_discrete(path):
        """
        Loads the positive class examples from the first 10 percent of the dataset.
        """
        data_train, data_test = load_data(path)

        # Gets rid of any background noise examples i.e. class label 0.
        data_train = data_train[data_train[data_train.columns[0]] == 1]
        data_train = data_train.drop(data_train.columns[0], axis=1)
        data_test = data_test[data_test[data_test.columns[0]] == 1]
        data_test = data_test.drop(data_test.columns[0], axis=1)
        # Because the data_ set is messed up!
        data_test = data_test.drop(data_test.columns[-1], axis=1)

        return data_train, data_test

    def load_data_no_discrete_normalised(path):

        data_train, data_test = load_data_no_discrete(path)
        mu = data_train.mean()
        s = data_train.std()
        data_train = (data_train - mu) / s
        data_test = (data_test - mu) / s

        return data_train, data_test

    def load_data_no_discrete_normalised_as_array(path):

        data_train, data_test = load_data_no_discrete_normalised(path)
        data_train, data_test = data_train.values, data_test.values

        i = 0
        # Remove any features that have too many re-occurring real values.
        features_to_remove = []
        for feature in data_train.T:
            c = Counter(feature)
            max_count = np.array([v for k, v in sorted(c.items())])[0]
            if max_count > 5:
                features_to_remove.append(i)
            i += 1
        data_train = data_train[
            :,
            np.array(
                [i for i in range(data_train.shape[1]) if i not in features_to_remove]
            ),
        ]
        data_test = data_test[
            :,
            np.array(
                [i for i in range(data_test.shape[1]) if i not in features_to_remove]
            ),
        ]

        N = data_train.shape[0]
        N_validate = int(N * 0.1)
        data_validate = data_train[-N_validate:]
        data_train = data_train[0:-N_validate]

        return data_train, data_validate, data_test

    path = os.path.join(data_root, "raw/hepmass")
    return load_data_no_discrete_normalised_as_array(path)


def preprocess_miniboone(
    data_root: str = "data",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Preprocesses the Miniboone dataset.

    Args:
        data_root: The directory where the data is stored.

    Returns:
        A tuple with the train, validation, and test partitions of the data.
    """

    def load_data(root_path):
        data = np.load(root_path)
        N_test = int(0.1 * data.shape[0])
        data_test = data[-N_test:]
        data = data[0:-N_test]
        N_validate = int(0.1 * data.shape[0])
        data_validate = data[-N_validate:]
        data_train = data[0:-N_validate]

        return data_train, data_validate, data_test

    def load_data_normalised(root_path):
        data_train, data_validate, data_test = load_data(root_path)
        data = np.vstack((data_train, data_validate))
        mu = data.mean(axis=0)
        s = data.std(axis=0)
        data_train = (data_train - mu) / s
        data_validate = (data_validate - mu) / s
        data_test = (data_test - mu) / s

        return data_train, data_validate, data_test

    path = os.path.join(data_root, "raw/miniboone/data.npy")
    return load_data_normalised(path)


def preprocess_bsds(
    data_root: str = "data",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Preprocesses the BSDS300 dataset.

    Args:
        data_root: The directory where the data is stored.

    Returns:
        A tuple with the train, validation, and test partitions of the data.
    """
    path = os.path.join(data_root, "raw/BSDS300/BSDS300.hdf5")
    file = h5py.File(path, "r")
    return file["train"], file["validation"], file["test"]


def preprocess_uci_data(data_root="data"):
    """Preprocesses all UCI datasets.

    Args:
        data_root: The directory where the data is stored.

    Returns:
        None.
    """
    preprocess_dict = {
        "gas": preprocess_gas,
        "power": preprocess_power,
        "hepmass": preprocess_hepmass,
        "miniboone": preprocess_miniboone,
        "bsds": preprocess_bsds,
    }

    for dataset, preprocess_fn in preprocess_dict.items():
        train, val, test = preprocess_fn(data_root)
        splits = (("train", train), ("val", val), ("test", test))
        output_dir = os.path.join(data_root, "processed", dataset)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        for split in splits:
            name, data = split
            file = os.path.join(output_dir, "{}.npy".format(name))
            np.save(file, data)


def download_and_extract(data_root="data"):
    """Downloads and extracts the raw UCI data.

    Args:
        data_root: The directory where the data will be stored.

    Returns:
        None.
    """
    if not os.path.exists(data_root):
        os.makedirs(data_root)
    filename = os.path.join(data_root, "data.tar.gz")

    def reporthook(tbar):
        last_block = [0]

        def update_to(block=1, block_size=1, tbar_size=None):
            if tbar_size is not None:
                tbar.total = tbar_size
            tbar.update((block - last_block[0]) * block_size)
            last_block[0] = block

        return update_to

    with tqdm(unit="B", unit_scale=True, unit_divisor=1024, miniters=1) as tbar:
        tbar.set_description("Downloading datasets...")
        request.urlretrieve(
            url="https://zenodo.org/record/1161203/files/data.tar.gz?download=1",
            filename=filename,
            reporthook=reporthook(tbar),
        )
        tbar.set_description("Finished downloading.")

    print("\nExtracting datasets...")
    with tarfile.open(filename, "r:gz") as file:
        print(filename)
        file.extractall(path=data_root)
    print("Finished extraction.\n")

    print("Removing zipped data...")
    os.remove(filename)
    os.rename(os.path.join(data_root, "data"), os.path.join(data_root, "raw"))
    print("Zipped data removed.\n")

    shutil.rmtree(os.path.join(data_root, "data/cifar10"))
    print("CIFAR-10 removed.\n")


def download_preprocess_data(data_root="data"):
    """Downloads and preprocesses the UCI data.

    Args:
        data_root: The directory where the data will be stored.

    Returns:
        None.
    """
    download_and_extract(data_root)
    print("Preprocessing data...")
    preprocess_uci_data(data_root)
    print("Data processed.")


@click.command()
def main():
    """Downloads and preprocesses the UCI data."""
    data_dir = pathlib.Path(__file__).parent.parent.parent.joinpath("data")

    query = (
        "> Running this script will download and preprocess UCI and BSDS300 data.\n"
        "> The zipped download is 817MB in size, and 1.6GB once unzipped.\n"
        "> The download includes CIFAR-10, but it will be removed by the script.\n"
        "> After extraction, this script will also delete the zipped download.\n"
        "> Do you wish to download the data? [Y/n] "
    )
    response = input(query)

    if response in ["y", "Y"]:
        download_preprocess_data(data_dir)
    elif response not in ["n", "N"]:
        print("Response not understood.")


if __name__ == "__main__":
    main()
