import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

TRAIN_URL = (
    "https://drive.google.com/uc?id=14CJDG_EWpocULPFWde_eLJ_ixlavm3L-&export=download"
)
VAL_URL = (
    "https://drive.google.com/uc?id=1LH7d70p3oScUXNLiWXN1ZzDT9oc3i5mf&export=download"
)
TEST_URL = (
    "https://drive.google.com/uc?id=1QsJ9RW_-VdMIdM87i-_y3VSmBlibcdr1&export=download"
)


class Miniboone(tfds.core.GeneratorBasedBuilder):
    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {
        "1.0.0": "Initial release.",
    }

    def _info(self) -> tfds.core.DatasetInfo:
        return tfds.core.DatasetInfo(
            builder=self,
            description="The UCI Miniboone dataset for density estimation.",
            features=tfds.features.FeaturesDict(
                {"features": tfds.features.Tensor(shape=(43,), dtype=tf.float32)}
            ),
            supervised_keys=None,
            homepage="https://archive.ics.uci.edu/ml/datasets/MiniBooNE+particle+identification",
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
