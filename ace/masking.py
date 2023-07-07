from abc import ABC, abstractmethod

import numpy as np
import tensorflow as tf


def enumerate_mask(num_features, mask_idx):
    assert mask_idx < 2**num_features
    binary = np.array([int(i) for i in bin(mask_idx)[2:].zfill(num_features)])
    return binary #.astype("float32")

class MaskGenerator(ABC):
    def __init__(
        self,
        seed=None,
        dtype=np.float32,
    ):
        self._rng = np.random.RandomState(seed=seed)
        self._dtype = dtype

    def __call__(self, shape):
        return self.call(np.asarray(shape)).astype(self._dtype)

    @abstractmethod
    def call(self, shape):
        pass


class FixedMaskGenerator(MaskGenerator):
    def __init__(self, mask, **kwargs):
        super().__init__(**kwargs)

        self.mask = np.array(mask).astype(self._dtype)

    def call(self, shape):
        return np.stack([np.squeeze(self.mask)] * shape[0])

    # def __call__(self, shape):
    #     return self.call(shape)

class MostlyFixedMaskGenerator(MaskGenerator):
    def __init__(self, mask, **kwargs):
        super().__init__(**kwargs)

        self.mask = np.array(mask).astype(self._dtype)

    def call(self, shape):

        if np.random.random() < 0.01:
            return self._rng.binomial(1, 0.5, size=shape)

        return np.stack([np.squeeze(self.mask)] * shape[0])


class BatchFixedMaskGenerator(MaskGenerator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, shape):

        mask = self._rng.binomial(1, 0.5, size=(1, shape[1]))

        mask_stack = np.stack([np.squeeze(mask)] * shape[0])

        return mask_stack
    

class DualMaskGenerator(MaskGenerator):
    def __init__(self, masks, **kwargs):
        super().__init__(**kwargs)
        self.masks = np.array(masks)
        # self.masks = self._rng.binomial(1, 0.5, size=mask.shape)

    def call(self, shape):

        mask_choice = np.round(np.random.random()).astype(int)

        mask_stack = np.stack([np.squeeze(self.masks[mask_choice])] * shape[0])

        return mask_stack

class BernoulliMaskGenerator(MaskGenerator):
    def __init__(self, p=0.5, **kwargs):
        super().__init__(**kwargs)

        self.p = p

    def call(self, shape):
        return self._rng.binomial(1, self.p, size=shape)


class UniformMaskGenerator(MaskGenerator):
    def call(self, shape):
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


def get_add_mask_fn(mask_generator):
    def fn(x):
        [mask] = tf.py_function(mask_generator, [tf.shape(x)], [x.dtype])
        # This reshape ensures that mask shape is not unknown.
        mask = tf.reshape(mask, tf.shape(x))
        return x, mask

    return fn


def get_add_marginal_masks_fn(marginal_dims):
    def fn(x):
        missing = tf.greater_equal(tf.range(tf.shape(x)[-1]), marginal_dims)
        missing = tf.cast(missing, x.dtype)
        return x, tf.zeros_like(x), missing

    return fn
