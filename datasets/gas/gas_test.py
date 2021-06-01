import tensorflow_datasets as tfds

import gas


class GasTest(tfds.testing.DatasetBuilderTestCase):
    DATASET_CLASS = gas.Gas
    SPLITS = {
        "train": 3,
        "val": 2,
        "test": 2,
    }

    DL_EXTRACT_RESULT = {
        "train": "train.txt",
        "val": "val.txt",
        "test": "test.txt",
    }


if __name__ == "__main__":
    tfds.testing.test_main()
