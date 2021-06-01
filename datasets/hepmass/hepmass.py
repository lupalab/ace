import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

TRAIN_URL = (
    "https://drive.google.com/uc?id=13PF7GfAhKy1WFgoFodpFlHEMK4ImDQsn&export=download"
)
VAL_URL = (
    "https://drive.google.com/uc?id=13w__3XmdCMSuXO3io9sKdxclefXV5GQP&export=download"
)
TEST_URL = (
    "https://drive.google.com/uc?id=1cgrGD-915n52buzHoMyaH9gRq1th-BDF&export=download"
)


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
        path = dl_manager.download(
            {
                "train": TRAIN_URL,
                "val": VAL_URL,
                "test": TEST_URL,
            }
        )

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
