import os

import gdown
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

TRAIN_ID = "1YKOxuYRC-79h9ZUHQ0d7v1bi9ozKE608"
VAL_ID = "1ySm47-7yvoQOHi3_Gbvme6vD2hAzt3Yq"
TEST_ID = "1ZoqOzNzfSe33cWkjKYeUuJ9C57Nf2YlO"


class BSDS(tfds.core.GeneratorBasedBuilder):
    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {
        "1.0.0": "Initial release.",
    }

    def _info(self) -> tfds.core.DatasetInfo:
        return tfds.core.DatasetInfo(
            builder=self,
            description="The BSDS300 dataset for density estimation.",
            features=tfds.features.FeaturesDict(
                {"features": tfds.features.Tensor(shape=(63,), dtype=tf.float32)}
            ),
            supervised_keys=None,
            homepage="https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/",
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
        file = tf.io.gfile.GFile(path)
        data = np.loadtxt(file, np.float32)

        for i, x in enumerate(data):
            yield i, dict(features=x)
