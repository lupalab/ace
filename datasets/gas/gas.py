import os

import gdown
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

TRAIN_ID = "1rhhLpGY5-5YZ-GQaZq_FniAMK9cqvxW7"
VAL_ID = "1y4TeXr5WhMWDZIwHh4iLIVj-eV3MgwBB"
TEST_ID = "17aRcAFtnYg6SVtGHksDZCQ5mgJwvve6L"


class Gas(tfds.core.GeneratorBasedBuilder):
    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {
        "1.0.0": "Initial release.",
    }

    def _info(self) -> tfds.core.DatasetInfo:
        return tfds.core.DatasetInfo(
            builder=self,
            description="The UCI Power dataset for density estimation.",
            features=tfds.features.FeaturesDict(
                {"features": tfds.features.Tensor(shape=(8,), dtype=tf.float32)}
            ),
            supervised_keys=None,
            homepage="https://archive.ics.uci.edu/ml/datasets/gas+sensor+array+under+dynamic+gas+mixtures",
            citation=None,
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        path = {
            "train": gdown.download(
                output=os.path.join(dl_manager.download_dir, "train.txt"), id=TRAIN_ID
            ),
            "val": gdown.download(
                output=os.path.join(dl_manager.download_dir, "val.txt"), id=VAL_ID
            ),
            "test": gdown.download(
                output=os.path.join(dl_manager.download_dir, "test.txt"), id=TEST_ID
            ),
        }

        return {
            "train": self._generate_examples(path["train"]),
            "val": self._generate_examples(path["val"]),
            "test": self._generate_examples(path["test"]),
        }

    def _generate_examples(self, path):
        data = np.loadtxt(path, np.float32)

        for i, x in enumerate(data):
            yield i, dict(features=x)
