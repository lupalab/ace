import tensorflow_datasets as tfds

import bsds


class BSDSTest(tfds.testing.DatasetBuilderTestCase):
    DATASET_CLASS = bsds.BSDS
    SPLITS = {
        "train": 3,
        "val": 3,
        "test": 3,
    }

    DL_EXTRACT_RESULT = {
        "train": "train.txt",
        "val": "val.txt",
        "test": "test.txt",
    }


if __name__ == "__main__":
    tfds.testing.test_main()
