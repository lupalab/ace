import os

import gdown
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

TRAIN_ID = "13PF7GfAhKy1WFgoFodpFlHEMK4ImDQsn"
VAL_ID = "13w__3XmdCMSuXO3io9sKdxclefXV5GQP"
TEST_ID = "1cgrGD-915n52buzHoMyaH9gRq1th-BDF"


class Hepmass(tfds.core.GeneratorBasedBuilder):
    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {
        "1.0.0": "Initial release.",
    }

    def _info(self) -> tfds.core.DatasetInfo:
        return tfds.core.DatasetInfo(
            builder=self,
            description="The UCI Hepmass dataset for density estimation.",
            features=tfds.features.FeaturesDict(
                {"features": tfds.features.Tensor(shape=(21,), dtype=tf.float32)}
            ),
            supervised_keys=None,
            homepage="https://archive.ics.uci.edu/ml/datasets/HEPMASS",
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
