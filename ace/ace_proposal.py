from typing import NamedTuple, Optional, Union, Tuple

import gin
import numpy as np
import tensorflow as tf

from tensorflow.keras import layers as tfl

from tensorflow.keras.regularizers import Regularizer
from tensorflow.keras import backend

from ace.networks import proposal_network, energy_network

class BaselineRegularizer(Regularizer):
    def __init__(self, penalty, baseline=None):

        self.penalty = penalty

        if baseline is not None:
            self.baseline = backend.cast_to_floatx(baseline)
        else:
            self.baseline = None

    def set_baseline(self, baseline):
        self.baseline = backend.cast_to_floatx(baseline)

    def __call__(self, x):
        return tf.keras.losses.MeanSquaredError(self.baseline, x) * self.penalty


class ACEOutput(NamedTuple):
    """Contains the outputs of a forward pass of an ACEModel."""

    # energy_ll: tf.Tensor
    # unnormalized_energy_ll: tf.Tensor
    proposal_ll: tf.Tensor
    # log_ratios: tf.Tensor

    proposal_samples: tf.Tensor
    # proposal_samples_log_ratios: tf.Tensor
    # log_normalizers: tf.Tensor

    proposal_mean: tf.Tensor
    # energy_mean: tf.Tensor


@gin.configurable(denylist=["num_features"])
class ACEModel(tf.keras.Model):
    """An Arbitrary Conditioning with Energy model.

    This class encapsulates an ACE model, which can be trained to perform arbitrary
    conditional density estimation. That is, this model can simultaneously estimate the
    distribution p(x_u | x_o) for all possible subsets of unobserved features x_u and
    observed features x_o.

    This implementation is for continuous data.

    Args:
        num_features: The dimensionality of the data that this model will be used with.
        context_units: The dimensionality of the context vectors that are
            outputted by the proposal network and passed into the energy network.
        mixture_components: The number of components in each mixture of
            Gaussians proposal distribution.
        proposal_residual_blocks: The number of residual blocks in the proposal
           network.
        proposal_hidden_units: The number of units in each hidden layer of the
            proposal network.
        energy_residual_blocks: The number of residual blocks in the energy network.
        energy_hidden_units: The number of units in each hidden layer of the
            energy network.
        activation: The activation function.
        dropout: The dropout rate.
        energy_clip: The maximum allowed value of energies that are outputted by the
            energy network. This clipping is used to improve training stability.
        training_importance_samples: The number of importance samples that are used
           when training with `.fit()`.
        energy_regularization: The coefficient of the energy regularization loss term
           when training with `.fit()`.
    """

    def __init__(
        self,
        num_features: int,
        context_units: int = 64,
        mixture_components: int = 10,
        proposal_residual_blocks: int = 4,
        proposal_hidden_units: int = 512,
        energy_residual_blocks: int = 4,
        energy_hidden_units: int = 128,
        activation: str = "relu",
        dropout: float = 0.0,
        energy_clip: float = 30.0,
        training_importance_samples: int = 20,
        energy_regularization: float = 0.0,
        proposal_regularization: float = 0.01,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self._config = locals()
        del self._config["self"]
        del self._config["__class__"]
        self._config.update(self._config.pop("kwargs"))

        self._num_features = num_features
        self._context_units = context_units
        self._training_importance_samples = training_importance_samples
        self._energy_regularization = energy_regularization

        self.finetune_kernel_reg = BaselineRegularizer(penalty=proposal_regularization)
        self.finetune_bias_reg = BaselineRegularizer(penalty=proposal_regularization)

        self.finetune_layer = tfl.Dense(
            num_features * (3 * mixture_components + context_units), 
            kernel_regularizer=self.finetune_kernel_reg,
            bias_regularizer=self.finetune_bias_reg,
            name="finetune_linear")

        self._proposal_network = proposal_network(
            num_features,
            context_units,
            mixture_components,
            proposal_residual_blocks,
            proposal_hidden_units,
            activation,
            dropout,
            self.finetune_layer,
            name="proposal_network",
        )

        # self._energy_network = energy_network(
        #     num_features,
        #     context_units,
        #     energy_residual_blocks,
        #     energy_hidden_units,
        #     activation,
        #     dropout,
        #     energy_clip,
        #     name="energy_network",
        # )

        self._metrics = [
            tf.keras.metrics.Mean(name="loss"),
            # tf.keras.metrics.Mean(name="energy_ll"),
            tf.keras.metrics.Mean(name="proposal_ll"),
        ]
        # if self._energy_regularization != 0.0:
        #     self._metrics.append(tf.keras.metrics.Mean(name="energy_reg_loss"))

        self._alpha = tf.Variable(1.0, trainable=False, dtype=tf.float32)

        # Create model weights
        fake_data = tf.zeros((1, num_features))
        self([fake_data, fake_data], num_importance_samples=1)

    def _process_inputs(self, x, observed_mask, missing_mask):
        observed_mask = tf.cast(observed_mask, x.dtype)
        query = 1.0 - observed_mask

        if missing_mask is not None:
            missing_mask = tf.cast(missing_mask, x.dtype)
            query *= 1.0 - missing_mask
            observed_mask *= 1.0 - missing_mask

        x_o = x * observed_mask
        x_u = x * query

        return x_o, x_u, observed_mask, query

    # def _get_energy_inputs(
    #     self,
    #     x_u,
    #     proposal_samples,
    #     num_importance_samples,
    #     context,
    #     selected_features,
    # ):
    #     x_u_and_samples = tf.concat(
    #         [tf.expand_dims(x_u, 1), tf.cast(proposal_samples, x_u.dtype)], axis=1
    #     )
    #     u_i = tf.broadcast_to(
    #         tf.range(self._num_features, dtype=tf.int32),
    #         [tf.shape(x_u)[0], 1 + num_importance_samples, self._num_features],
    #     )

    #     if selected_features is not None:
    #         x_u_and_samples = tf.gather(
    #             x_u_and_samples, selected_features, batch_dims=1, axis=2
    #         )
    #         u_i = tf.gather(u_i, selected_features, batch_dims=1, axis=2)
    #         context = tf.gather(context, selected_features, batch_dims=1, axis=1)

    #     tiled_context = tf.tile(
    #         tf.expand_dims(context, 1), [1, 1 + num_importance_samples, 1, 1]
    #     )

    #     x_u_i = tf.reshape(x_u_and_samples, [-1])
    #     u_i = tf.reshape(u_i, [-1])
    #     tiled_context = tf.reshape(tiled_context, [-1, self._context_units])

    #     return x_u_i, u_i, tiled_context

    def call(
        self,
        inputs,
        missing_mask=None,
        num_importance_samples=10,
        training=None,
        selected_features=None,
    ) -> ACEOutput:
        """Forward pass through the ACE model.

        Args:
            inputs: A list that contains the following:
                x: A tensor of shape `[batch_size, data_dim]` that contains the values
                    of the features, both observed and unobserved.
                observed_mask: A tensor of shape `[batch_size, data_dim]` that is 1
                    for features which are observed and 0 otherwise.
            missing_mask: Optional. A tensor of shape `[batch_size, data_dim]`
                that is 1 for features which are missing and 0 otherwise.
            training: A boolean indicating whether or not training mode should be used.
            num_importance_samples: The number of importance samples to use.
            selected_features: A rank 1 or rank 2 integer tensor that specifies which
                features should be processed by the energy network. By default, all
                features are batched together and sent through the energy network. This
                is necessary during training due to arbitrarily-sized observed sets,
                but it is computationally wasteful. During inference, we often only need
                energies for a specific dimension at a time. If this tensor has the
                shape `[num_inds]` then energies will only be computed for the specified
                indices for all of the instances. If this tensor has the
                shape `[batch_size, num_inds]`, then the same is true, but different
                indices can be specified for different instances. If this argument is
                provided, the shapes of the outputs will be affected accordingly.

        Returns:
            An ACEOutput tuple.
        """
        x_o, x_u, observed_mask, query = self._process_inputs(
            inputs[0], inputs[1], missing_mask
        )

        proposal_dist, context = self._proposal_network(
            [x_o, observed_mask], training=training
        )

        proposal_ll = proposal_dist.log_prob(tf.cast(x_u, tf.float32))
        proposal_ll *= tf.cast(query, tf.float32)

        proposal_samples = tf.stop_gradient(
            proposal_dist.sample(num_importance_samples)
        )
        # proposal_samples_proposal_ll = tf.stop_gradient(
        #     proposal_dist.log_prob(proposal_samples)
        # )
        proposal_samples = tf.transpose(proposal_samples, [1, 0, 2])
        # proposal_samples_proposal_ll = tf.transpose(
        #     proposal_samples_proposal_ll, [1, 0, 2]
        # )
        proposal_samples *= tf.expand_dims(tf.cast(query, tf.float32), 1)
        # proposal_samples_proposal_ll *= tf.expand_dims(tf.cast(query, tf.float32), 1)

        proposal_mean = proposal_dist.mean() * tf.cast(query, tf.float32)

        # x_u_i, u_i, tiled_context = self._get_energy_inputs(
        #     x_u,
        #     proposal_samples,
        #     num_importance_samples,
        #     context,
        #     selected_features,
        # )
        # negative_energies = self._energy_network([x_u_i, u_i, tiled_context])

        # negative_energies = tf.reshape(
        #     negative_energies,
        #     [
        #         -1,
        #         1 + num_importance_samples,
        #         self._num_features
        #         if selected_features is None
        #         else selected_features.shape[-1],
        #     ],
        # )

        if selected_features is not None:
            query = tf.gather(query, selected_features, batch_dims=1, axis=1)
            proposal_mean = tf.gather(
                proposal_mean, selected_features, batch_dims=1, axis=1
            )
            proposal_ll = tf.gather(
                proposal_ll, selected_features, batch_dims=1, axis=1
            )
            proposal_samples_proposal_ll = tf.gather(
                proposal_samples_proposal_ll,
                selected_features,
                batch_dims=1,
                axis=2,
            )
            proposal_samples = tf.gather(
                proposal_samples,
                selected_features,
                batch_dims=1,
                axis=2,
            )

        # negative_energies *= tf.expand_dims(query, 1)
        # negative_energies = tf.cast(negative_energies, tf.float32)
        query = tf.cast(query, tf.float32)

        # unnorm_energy_ll = negative_energies[:, 0]
        # proposal_samples_unnorm_energy_ll = negative_energies[:, 1:]

        # log_ratios = unnorm_energy_ll - proposal_ll
        # proposal_samples_log_ratios = (
        #     proposal_samples_unnorm_energy_ll - proposal_samples_proposal_ll
        # )

        # log_normalizers = (
        #     tf.reduce_logsumexp(proposal_samples_log_ratios, axis=1)
        #     - tf.math.log(tf.cast(num_importance_samples, tf.float32))
        # ) * tf.cast(query, tf.float32)

        # energy_ll = unnorm_energy_ll - log_normalizers

        # is_weights = tf.nn.softmax(proposal_samples_log_ratios, axis=1)
        # energy_mean = tf.reduce_sum(is_weights * proposal_samples, axis=1)

        proposal_samples *= tf.expand_dims(query, 1)
        # proposal_samples_log_ratios *= tf.expand_dims(query, 1)
        proposal_mean *= query
        # energy_mean *= query

        return ACEOutput(
            # energy_ll,
            # unnorm_energy_ll,
            proposal_ll,
            # log_ratios,
            proposal_samples,
            # proposal_samples_log_ratios,
            # log_normalizers,
            proposal_mean,
            # energy_mean,
        )

    def train_step(self, data):
        x, b, *m = data
        with tf.GradientTape() as tape:
            ace_output = self(
                [x, b],
                *m,
                training=True,
                num_importance_samples=self._training_importance_samples,
            )
            # energy_ll = tf.reduce_sum(ace_output.energy_ll, -1)
            proposal_ll = tf.reduce_sum(ace_output.proposal_ll, -1)

            # energy_ll = tf.nn.compute_average_loss(energy_ll)
            proposal_ll = tf.nn.compute_average_loss(proposal_ll)

            # if self._energy_regularization != 0.0:
            #     energy_reg_loss = tf.nn.compute_average_loss(
            #         tf.losses.mse(
            #             ace_output.energy_ll,
            #             tf.stop_gradient(ace_output.proposal_ll),
            #         ),
            #     )
            # else:
            #     energy_reg_loss = 0.0

            loss = (
                -(proposal_ll) # + (self._alpha * energy_ll))
                # + self._energy_regularization * energy_reg_loss
            )

        self.optimizer.minimize(loss, self.trainable_variables, tape=tape)

        self._metrics[0].update_state(loss)
        # self._metrics[1].update_state(energy_ll)
        self._metrics[1].update_state(proposal_ll)
        # if self._energy_regularization != 0.0:
        #     self._metrics[3].update_state(energy_reg_loss)

        return {m.name: m.result() for m in self._metrics}

    def test_step(self, data):
        ace_output = self(
            data,
            training=False,
            num_importance_samples=self._training_importance_samples,
        )

        # energy_ll = tf.reduce_sum(ace_output.energy_ll, -1)
        proposal_ll = tf.reduce_sum(ace_output.proposal_ll, -1)

        # energy_ll = tf.nn.compute_average_loss(energy_ll)
        proposal_ll = tf.nn.compute_average_loss(proposal_ll)

        # self._metrics[1].update_state(energy_ll)
        self._metrics[1].update_state(proposal_ll)

        return {m.name: m.result() for m in self._metrics[1:]}

    @property
    def metrics(self):
        return self._metrics

    def get_config(self):
        return self._config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def _single_permutation_ll(
        self,
        x,
        observed_mask,
        missing_mask=None,
        num_importance_samples=20,
        **map_fn_kwargs,
    ):
        if missing_mask is None:
            missing_mask = tf.broadcast_to(0.0, tf.shape(observed_mask))

        query = (1.0 - observed_mask) * (1.0 - missing_mask)
        observed_mask *= 1.0 - missing_mask

        expanded_x, expanded_observed_mask, selected_features = tf.map_fn(
            _get_autoregressive_ll_batch,
            [x, observed_mask, query],
            fn_output_signature=(
                tf.RaggedTensorSpec((None, x.shape[-1]), x.dtype),
                tf.RaggedTensorSpec((None, x.shape[-1]), observed_mask.dtype),
                tf.RaggedTensorSpec((None, 1), tf.int64),
            ),
            **map_fn_kwargs,
        )

        query_total = tf.math.count_nonzero(query)
        batch_x = expanded_x.merge_dims(0, 1).to_tensor(
            shape=(query_total, self._num_features)
        )
        batch_observed_mask = expanded_observed_mask.merge_dims(0, 1).to_tensor()
        batch_selected_features = tf.expand_dims(selected_features.merge_dims(0, 2), 1)
        batch_selected_features = tf.cast(batch_selected_features, tf.int32)

        outputs = self(
            [batch_x, batch_observed_mask],
            num_importance_samples=num_importance_samples,
            selected_features=batch_selected_features,
        )

        u_count = tf.math.count_nonzero(query, axis=-1)

        # energy_ll = tf.RaggedTensor.from_row_lengths(
        #     tf.squeeze(outputs.energy_ll, -1), u_count
        # )
        proposal_ll = tf.RaggedTensor.from_row_lengths(
            tf.squeeze(outputs.proposal_ll, 1), u_count
        )

        # energy_ll = tf.reduce_sum(energy_ll, axis=-1)
        proposal_ll = tf.reduce_sum(proposal_ll, axis=-1)

        return None, proposal_ll # energy_ll, proposal_ll

    def log_prob(
        self,
        x: Union[np.ndarray, tf.Tensor],
        observed_mask: Union[np.ndarray, tf.Tensor],
        missing_mask: Optional[Union[np.ndarray, tf.Tensor]] = None,
        num_importance_samples: int = 20,
        num_permutations: Optional[int] = None,
        **map_fn_kwargs,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """Computes conditional log-likelihoods.

        This method returns `log p(x_u | x_o)`, where `observed_mask` indicates which
        features are in `x_u` and `x_o`.

        Args:
            x: A tensor of shape `[batch_size, num_features]` that contains the values
                of the oberved and unobserved features.
            observed_mask: A binary tensor of shape `[batch_size, num_features]` that
                if 1 for features which are observed and 0 otherwise.
            missing_mask: Optional. A tensor of shape `[batch_size, num_features]` that
                is 1 for features which are missing and 0 otherwise. Missing features
                will be ignored by ACE.
            num_importance_samples: The number of importance samples to use when
                estimating normalizing constants.
            num_permutations: The number of random permutations of unobserved features
                of each instance to autoregressively compute log probabilities for.

        Returns:
            Two arrays with the energy log probabilities and the proposal log
            probabilities. If `num_permutations` is specified, each array has shape
            `[batch_size, num_permutations]`. Otherwise, each array has shape,
            `[batch_size]`.
        """
        if num_permutations is None:
            return self._single_permutation_ll(
                x,
                observed_mask,
                missing_mask,
                num_importance_samples,
                **map_fn_kwargs,
            )

        # ell_out = tf.TensorArray(x.dtype, size=num_permutations)
        pll_out = tf.TensorArray(x.dtype, size=num_permutations)

        for i in tf.range(num_permutations):
            ell, pll = self._single_permutation_ll(
                x,
                observed_mask,
                missing_mask,
                num_importance_samples,
                **map_fn_kwargs,
            )

            # ell_out = ell_out.write(i, ell)
            pll_out = pll_out.write(i, pll)

        # ell_out.close()
        pll_out.close()

        # energy_out = tf.transpose(ell_out.stack())
        energy_out = None
        proposal_out = tf.transpose(pll_out.stack())

        return energy_out, proposal_out

    def _dim_proposal_sample(self, x_o, observed_mask, index):
        proposal_dist, _ = self._proposal_network([x_o, observed_mask])
        samples = proposal_dist.sample()
        return samples[:, index]

    def _dim_energy_sample(self, x_o, observed_mask, index, num_resampling_samples):
        outputs = self(
            [x_o, observed_mask],
            num_importance_samples=num_resampling_samples,
            selected_features=tf.broadcast_to(index, (tf.shape(x_o)[0], 1)),
        )

        resampled_inds = tf.cast(tf.random.categorical(outputs.log_ratios, 1), tf.int32)
        winners = tf.gather(
            outputs.proposal_samples[..., 0], resampled_inds, batch_dims=1
        )
        return winners[:, 0]

    def _instance_autoregressive_sample(self, t):
        x_o, observed_mask, num_samples, num_resampling_samples, use_proposal = t
        u_inds = tf.where((1.0 - observed_mask) == 1)[:, 0]
        u_inds = tf.random.shuffle(u_inds)
        u_inds = tf.cast(u_inds, tf.int32)

        x_o = x_o * observed_mask
        curr_x_o = tf.tile(x_o[tf.newaxis, ...], [num_samples, 1])
        curr_observed_mask = tf.tile(observed_mask[tf.newaxis, ...], [num_samples, 1])

        for j in u_inds:
            if use_proposal:
                dim_samples = self._dim_proposal_sample(curr_x_o, curr_observed_mask, j)
            else:
                dim_samples = self._dim_energy_sample(
                    curr_x_o, curr_observed_mask, j, num_resampling_samples
                )

            update_inds = tf.stack(
                [tf.range(num_samples), tf.repeat(j, num_samples)], axis=1
            )

            curr_x_o = tf.tensor_scatter_nd_update(curr_x_o, update_inds, dim_samples)
            curr_observed_mask = tf.tensor_scatter_nd_update(
                curr_observed_mask, update_inds, tf.ones_like(dim_samples)
            )

        return curr_x_o

    def sample(
        self,
        x_o: Union[np.ndarray, tf.Tensor],
        observed_mask: Union[np.ndarray, tf.Tensor],
        num_samples: Optional[int] = None,
        num_resampling_samples: int = 20,
        use_proposal: bool = False,
        **map_fn_kwargs,
    ) -> tf.Tensor:
        """Samples the given conditional distribution.

        Args:
            x_o: A tensor of shape `[batch_size, num_features]` that contains the values
                of the oberved and unobserved features.
            observed_mask: A binary tensor of shape `[batch_size, num_features]` that
                if 1 for features which are observed and 0 otherwise.
            num_samples: The number of samples to draw for each instance.
            num_resampling_samples: The number of proposal samples for which importance
                weights are computed and a resampling is performed over. The higher
                this number is, the more closely the energy samples will approximate
                the true energy distribution.
            use_proposal: If True, samples will be drawn from the proposal distribution
                instead of the energy distribution.

        Returns:
            An array of shape `[batch_size, num_samples, num_features]`, unless
            `num_samples` is not specified, in which case the array will have size
            `[batch_size, num_features]`.
        """
        b = tf.shape(x_o)[0]
        num_samples_ = num_samples or 1

        samples = tf.map_fn(
            self._instance_autoregressive_sample,
            [
                x_o,
                observed_mask,
                tf.repeat(num_samples_, b),
                tf.repeat(num_resampling_samples, b),
                tf.repeat(use_proposal, b),
            ],
            fn_output_signature=tf.TensorSpec((num_samples_, self._num_features)),
            **map_fn_kwargs,
        )

        if num_samples is None:
            samples = tf.squeeze(samples, 1)

        return samples

    def impute(
        self,
        x_o: Union[np.ndarray, tf.Tensor],
        observed_mask: Union[np.ndarray, tf.Tensor],
        num_importance_samples: int = 20,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """Imputes unobserved values.

        Args:
            x_o: A tensor of shape `[batch_size, num_features]` that contains the values
                of the observed features.
            observed_mask: A binary tensor of shape `[batch_size, num_features]` that
                if 1 for features which are observed and 0 otherwise.
            num_importance_samples: The number of importance samples to use when
                estimating the mean of the energy distribution.

        Returns:
            Two arrays of shape `[batch_size, num_features]` that contains the imputed
            versions of `x_o` using the energy and proposal distributions.
        """
        x_o *= observed_mask
        outputs = self(
            [x_o, observed_mask], num_importance_samples=num_importance_samples
        )
        # energy_imputations = x_o + outputs.energy_mean
        proposal_imputations = x_o + outputs.proposal_mean
        energy_imputations = tf.zeros_like(proposal_imputations)
        return energy_imputations, proposal_imputations


def _get_autoregressive_ll_batch(t):
    x, observed_mask, query = t
    u_inds = tf.squeeze(tf.where(query), 1)
    u_inds = tf.random.shuffle(u_inds)
    mask = tf.math.cumsum(
        tf.one_hot(u_inds, x.shape[-1], dtype=observed_mask.dtype),
        axis=0,
        exclusive=True,
    )
    mask += tf.expand_dims(observed_mask, 0)
    x = tf.tile(x[tf.newaxis, :], [tf.shape(u_inds)[0], 1])

    return (
        tf.RaggedTensor.from_tensor(x),
        tf.RaggedTensor.from_tensor(mask),
        tf.RaggedTensor.from_tensor(tf.expand_dims(u_inds, 1)),
    )
