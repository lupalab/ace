from abc import ABC, abstractmethod
from typing import Optional, Union, Sequence

import numpy as np


class MaskGenerator(ABC):
    """Represents a generator that produces random binary masks.

    Args:
        seed: Optional. The random seed used by the generator.
        dtype: Optional. The data type of the generator.
    """

    def __init__(
        self, seed: Optional[int] = None, dtype: Union[str, object] = np.float32
    ):
        self._rng = np.random.RandomState(seed=seed)
        self._dtype = dtype

    def __call__(self, shape: Sequence[int]):
        return self.call(np.asarray(shape)).astype(self._dtype)

    @abstractmethod
    def call(self, shape: Sequence[int]) -> np.ndarray:
        """Creates a mask of the specified shape.

        Args:
            shape: The shape of the mask to generate.

        Returns:
            A binary ndarray with the specified shape.
        """
        pass


class UniformMaskGenerator(MaskGenerator):
    """A mask generator that creates uniform masks.

    This generator assumes that shapes will be of the form (batch_dim, data_dim).

    For each instance, the number of dimensions that will be "on" is first uniformly
    chosen at random. Then, that many dimensions are selected at random to be 1.
    """

    def call(self, shape: Sequence[int]) -> np.ndarray:
        assert len(shape) == 2, "expected shape of size (batch_dim, data_dim)"
        b, d = shape

        result = []
        for _ in range(b):
            q = self._rng.choice(d)
            inds = self._rng.choice(d, q, replace=False)
            mask = np.zeros(d)
            mask[inds] = 1
            result.append(mask)

        return np.vstack(result)


class BernoulliMaskGenerator(MaskGenerator):
    """A mask generator that creates bernoulli masks.

    This mask generator randomly sets each value to 1 with a probability of `p`.

    Args:
         p: The parameter of the Bernoulli distribution.
    """

    def __init__(self, p: float = 0.5, **kwargs):
        super().__init__(**kwargs)

        self.p = p

    def call(self, shape: Sequence[int]) -> np.ndarray:
        return self._rng.binomial(1, self.p, size=shape)


def get_mask_generator(
    mask_type: str, seed: Optional[int] = None, dtype: Union[str, object] = np.float32
):
    """Gets the specified mask generator.

    Args:
        mask_type: The type of mask generator to return, either "bernoulli"
            or "uniform".
        seed: Optional. The random seed used by the returned generator.
        dtype: Optional. The data type of the returned generator.

    Returns:
        A `MaskGenerator` of the requested type.
    """
    generators = {"bernoulli": BernoulliMaskGenerator, "uniform": UniformMaskGenerator}
    return generators[mask_type](seed=seed, dtype=dtype)
