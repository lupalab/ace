import os
from typing import Union, Optional, Tuple

import gin
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from ace.networks import ProposalNetwork, EnergyNetwork
from ace.utils import make_input_dict

tfd = tfp.distributions


def _sample_proposal(proposal_dist, num_samples, seed=None):
    # Sample from the proposal distribution and get logprobs of the samples
    proposal_samples = proposal_dist.sample(num_samples, seed=seed)
    proposal_log_prob_samples_proposal = proposal_dist.log_prob(proposal_samples)

    # Stop gradients to prevent backprop wrt proposal samples
    proposal_samples = tf.stop_gradient(tf.transpose(proposal_samples, [1, 0, 2]))
    proposal_log_prob_samples_proposal = tf.stop_gradient(
        tf.transpose(proposal_log_prob_samples_proposal, [1, 0, 2])
    )

    return proposal_samples, proposal_log_prob_samples_proposal


def _select_indices(x, indices, axis=1):
    if len(indices.shape) == 1:
        x = tf.gather(x, indices, axis=axis)
    elif len(indices.shape) == 2:
        x = tf.gather(x, indices, batch_dims=1, axis=axis)
    else:
        raise ValueError("indices must be rank 1 or 2")

    return x


@gin.configurable
class ACE(tf.keras.layers.Layer):
    """An Arbitrary Conditioning with Energy model.

    This class encapsulates an ACE model, which can be trained to perform arbitrary
    conditional density estimation. That is, this model can simultaneously estimate the
    distribution p(x_u | x_o) for all possible subsets of unobserved features x_u and
    observed features x_o.

    Args:
        proposal_comp_scale_min: The minimum allowed scale of the Gaussian components
            of the proposal distributions.
        num_proposal_mixture_comps: The number of components in each mixture of
            Gaussians proposal distribution.
        num_res_blocks_proposal: The number of residual blocks in the proposal network.
        num_hidden_units_proposal: The number of units in each hidden layer of the
            proposal network.
        activation_proposal: The activation function used in the proposal network.
        dropout_proposal: The dropout rate used in the proposal network. Optional.
        num_context_units_energy: The dimensionality of the context vectors that are
            outputted by the proposal network and passed into the energy network.
            If None, context vectors will not be used, and there will be no weight
            sharing between the networks.
        num_res_blocks_energy: The number of residual blocks in the energy network.
        num_hidden_units_energy: The number of units in each hidden layer of the
            energy network.
        activation_energy: The activation function used in the energy network.
        dropout_energy: The dropout rate used in the energy network. Optional.
        training_noise_scale: The scale of the Gaussian noise that is added to the
            model's input data during training.
        energy_clip: The maximum allowed value of energies that are outputted by the
            energy network. This clipping is used to improve training stability.
        name: The name of the model.
    """

    def __init__(
        self,
        proposal_comp_scale_min: float = 1e-3,
        num_proposal_mixture_comps: int = 20,
        num_res_blocks_proposal: int = 4,
        num_hidden_units_proposal: int = 512,
        activation_proposal: str = "relu",
        dropout_proposal: Optional[float] = None,
        num_context_units_energy: Optional[int] = 64,
        num_res_blocks_energy: int = 4,
        num_hidden_units_energy: int = 128,
        activation_energy: str = "relu",
        dropout_energy: Optional[float] = None,
        training_noise_scale: float = 0.001,
        energy_clip: float = 30,
        name: str = "ace",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self._proposal_comp_scale_min = proposal_comp_scale_min
        self._num_proposal_mixture_comps = num_proposal_mixture_comps
        self._num_context_units_energy = num_context_units_energy
        self._training_noise_scale = training_noise_scale
        self._energy_clip = energy_clip

        self._proposal_network = ProposalNetwork(
            self._num_proposal_mixture_comps,
            self._num_context_units_energy,
            num_res_blocks_proposal,
            num_hidden_units_proposal,
            activation_proposal,
            dropout_proposal,
        )

        self._energy_network = EnergyNetwork(
            num_res_blocks_energy,
            num_hidden_units_energy,
            activation_energy,
            dropout_energy,
        )

    def _process_inputs(self, inputs, training=False):
        x = inputs.get("x")
        observed_mask = inputs.get("observed_mask")
        missing_mask = inputs.get("missing_mask")

        if x is None:
            raise ValueError("ACE must be called with 'x' in the input dictionary.")
        if observed_mask is None:
            raise ValueError(
                "ACE must be called with 'observed_mask' in the input dictionary."
            )
        if missing_mask is None:
            missing_mask = tf.zeros_like(observed_mask)

        # Add small gaussian noise to data during training
        if training and self._training_noise_scale is not None:
            x += tf.random.normal(
                tf.shape(x), stddev=self._training_noise_scale, dtype=x.dtype
            )

        # Indicates which unobserved features are to be assessed
        query = (1.0 - observed_mask) * (1.0 - missing_mask)
        # Indicates which features are observed
        observed_mask *= 1.0 - missing_mask

        # Observed features
        x_o = x * observed_mask
        # Unobserved features
        x_u = x * query

        return x_o, x_u, observed_mask, query

    def _create_proposal_dist(self, proposal_outputs):
        if self._num_context_units_energy is not None:
            energy_context = proposal_outputs[..., : self._num_context_units_energy]
            proposal_params = proposal_outputs[..., self._num_context_units_energy :]
        else:
            energy_context = None
            proposal_params = proposal_outputs

        proposal_params = tf.cast(proposal_params, tf.float32)

        # Extract proposal distribution parameters from the network's output
        proposal_logits = proposal_params[..., : self._num_proposal_mixture_comps]
        proposal_means = proposal_params[
            ..., self._num_proposal_mixture_comps : 2 * self._num_proposal_mixture_comps
        ]
        proposal_scales = (
            tf.nn.softplus(proposal_params[..., 2 * self._num_proposal_mixture_comps :])
            + self._proposal_comp_scale_min
        )

        # Build the proposal distribution
        components_dist = tfd.Normal(loc=proposal_means, scale=proposal_scales)
        proposal_dist = tfd.MixtureSameFamily(
            mixture_distribution=tfd.Categorical(logits=proposal_logits),
            components_distribution=components_dist,
        )

        return proposal_dist, energy_context

    def _get_energy_net_inputs(
        self,
        num_importance_samples,
        mask,
        query,
        x_u,
        x_o,
        proposal_samples,
        energy_context,
        selected_indices=None,
    ):
        batch_size = tf.shape(x_o)[0]
        num_features = x_o.shape[-1]

        if selected_indices is not None:
            x_u = _select_indices(x_u, selected_indices)
            query = _select_indices(query, selected_indices)
            proposal_samples = _select_indices(
                proposal_samples, selected_indices, axis=2
            )

        data_dim = x_u.shape[-1]

        # [batch_size, num_samples, data_dim]
        x_u_samples = proposal_samples * query[:, tf.newaxis, :]
        # [batch_size, num_samples + 1, data_dim]
        x_u_cat_samples = tf.concat([x_u[:, tf.newaxis, :], x_u_samples], axis=1)
        # [batch_size * (num_samples + 1) * data_dim]
        x_u_i = tf.reshape(x_u_cat_samples, [-1])

        # [batch_size, num_samples + 1, num_features]
        x_o_tiled = tf.tile(x_o[:, tf.newaxis, :], [1, num_importance_samples + 1, 1])
        # [batch_size, (num_samples + 1), data_dim, num_features]
        x_o_tiled = tf.tile(x_o_tiled[..., tf.newaxis, :], [1, 1, data_dim, 1])
        # [batch_size * (num_samples + 1) * data_dim, num_features]
        x_o_tiled = tf.reshape(x_o_tiled, [-1, num_features])

        # [batch_size, num_samples + 1, num_features]
        mask_tiled = tf.tile(mask[:, tf.newaxis, :], [1, num_importance_samples + 1, 1])
        # [batch_size, (num_samples + 1), data_dim, num_features]
        mask_tiled = tf.tile(mask_tiled[..., tf.newaxis, :], [1, 1, data_dim, 1])
        # [batch_size * (num_samples + 1) * data_dim, num_features]
        mask_tiled = tf.reshape(mask_tiled, [-1, num_features])

        # [1, num_features]
        u_i = tf.range(num_features, dtype=tf.int32)[tf.newaxis, :]
        # [batch_size, num_features]
        u_i = tf.tile(u_i, [batch_size, 1])
        if selected_indices is not None:
            u_i = _select_indices(u_i, selected_indices)
        # [batch_size, num_samples + 1, data_dim]
        u_i = tf.tile(u_i[:, tf.newaxis, :], [1, 1 + num_importance_samples, 1])
        # [batch_size * (num_samples + 1) * data_dim]
        u_i = tf.reshape(u_i, [-1])

        if energy_context is not None:
            if selected_indices is not None:
                energy_context = _select_indices(energy_context, selected_indices)

            energy_context *= query[..., tf.newaxis]

            # [batch_size, num_samples + 1, data_dim, num_context_units]
            energy_context_tiled = tf.tile(
                energy_context[:, tf.newaxis, ...],
                [1, num_importance_samples + 1, 1, 1],
            )
            # [batch_size * (num_samples + 1) * data_dim, num_context_units]
            energy_context_tiled = tf.reshape(
                energy_context_tiled, [-1, self._num_context_units_energy]
            )
        else:
            energy_context_tiled = None

        return x_u_i, u_i, x_o_tiled, mask_tiled, energy_context_tiled

    def call(
        self,
        inputs,
        training=False,
        num_importance_samples=20,
        selected_indices=None,
        seed=None,
    ):
        """Forward pass through the ACE model.

        Args:
            inputs: A dictionary that contains the following:
                x: A tensor of shape `[batch_size, data_dim]` that contains the values
                    of the features, both observed and unobserved.
                observed_mask: A tensor of shape `[batch_size, data_dim]` that is 1
                    for features which are observed and 0 otherwise.
                missing_mask: Optional. A tensor of shape `[batch_size, data_dim]`
                    that is 1 for features which are missing and 0 otherwise.
            training: A boolean indicating whether or not training mode should be used.
            num_importance_samples: The number of importance samples to use.
            selected_indices: A rank 1 or rank 2 integer tensor that specifies which
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
            seed: The random seed used to generate samples and training noise.

        Returns:
            A dictionary with the following:
                energy_log_prob: The estimated normalized loglikelihoods for the energy
                    distributions.
                log_norm_constants_est: The estimated normalizing constants for the
                    energy distributions.
                unnorm_energy_log_prob: The unnormalized loglikelihoods produced by the
                    energy network.
                proposal_log_prob: The loglikelihoods for the proposal distribution.
                proposal_samples: The importance samples that were drawn from the
                    proposal distributions.
                proposal_log_prob_samples: The loglikelihoods of the proposal samples
                    for the proposal distributions.
                unnorm_energy_log_prob_samples: The unnormalized loglikelihoods of the
                    importance samples, from the energy network.
                proposal_mean: The means of the proposal distributions.
                energy_mean_est: The estimated means of the energy distributions.
                log_importance_weights: The log importance weights of the importance
                    samples.
        """
        x_o, x_u, mask, query = self._process_inputs(inputs, training=training)

        batch_size = tf.shape(x_o)[0]

        # Pass through the proposal network to get proposal parameters
        proposal_outputs = self._proposal_network([x_o, mask], training=training)
        # Create the proposal distributions
        proposal_dist, energy_context = self._create_proposal_dist(proposal_outputs)

        # Proposal log_probs of the input data
        proposal_log_prob = proposal_dist.log_prob(tf.cast(x_u, tf.float32)) * tf.cast(
            query, tf.float32
        )

        # Draw samples from the proposal distributions, and get their log probs
        proposal_samples, proposal_log_prob_samples = _sample_proposal(
            proposal_dist, num_importance_samples, seed=seed
        )
        proposal_log_prob_samples *= tf.cast(query[:, tf.newaxis, :], tf.float32)

        # Get the means of the proposal distributions
        proposal_mean = proposal_dist.mean() * tf.cast(query, tf.float32)

        # Construct the inputs to the energy network
        energy_net_inputs = self._get_energy_net_inputs(
            num_importance_samples,
            mask,
            query,
            x_u,
            x_o,
            tf.cast(proposal_samples, x_o.dtype),
            energy_context,
            selected_indices=selected_indices,
        )

        # Pass through the energy network to get energies
        # [batch_size * (num_samples + 1) * data_dim]
        energy_net_outputs = self._energy_network(energy_net_inputs, training=training)
        # [batch_size, num_samples + 1, data_dim]
        energy_net_outputs = tf.reshape(
            energy_net_outputs, [batch_size, num_importance_samples + 1, -1]
        )

        if selected_indices is not None:
            query = _select_indices(query, selected_indices)
            proposal_mean = _select_indices(proposal_mean, selected_indices)
            proposal_log_prob_samples = _select_indices(
                proposal_log_prob_samples, selected_indices, axis=2
            )
            proposal_log_prob = _select_indices(proposal_log_prob, selected_indices)
            proposal_samples = _select_indices(
                proposal_samples, selected_indices, axis=2
            )

        # Zero out energies of observed features
        energy_net_outputs *= query[:, tf.newaxis, :]

        # Enforce upper bound on the energies. The network is actually outputting
        # the negative energies, which is why the lower end is being clipped here.
        energy_net_outputs = tf.clip_by_value(energy_net_outputs, -self._energy_clip, 0)

        # Ensure all outputs are float32 even if mixed precision is on
        energy_net_outputs = tf.cast(energy_net_outputs, tf.float32)

        # Separate energies for the input data and the proposal samples.
        unnorm_energy_log_prob = energy_net_outputs[:, 0, :]
        unnorm_energy_log_prob_samples = energy_net_outputs[:, 1:, :]

        # Estimate normalizing constants with importance sampling
        log_norm_constants_est = tf.reduce_logsumexp(
            unnorm_energy_log_prob_samples - proposal_log_prob_samples, axis=1
        ) - tf.math.log(tf.cast(num_importance_samples, tf.float32))

        # Normalize the unnormalized log probabilities
        energy_log_prob = unnorm_energy_log_prob - log_norm_constants_est

        # Compute log importance weights for the proposal samples
        log_ratios = unnorm_energy_log_prob_samples - proposal_log_prob_samples

        # Estimate the mean of the energy distributions the importance samples
        weights = tf.nn.softmax(log_ratios, axis=1)
        energy_mean_est = tf.reduce_sum(weights * proposal_samples, axis=1)

        return {
            "energy_log_prob": energy_log_prob,
            "log_norm_constants_est": log_norm_constants_est,
            "unnorm_energy_log_prob": unnorm_energy_log_prob,
            "proposal_log_prob": proposal_log_prob,
            "proposal_samples": proposal_samples,
            "proposal_log_prob_samples": proposal_log_prob_samples,
            "unnorm_energy_log_prob_samples": unnorm_energy_log_prob_samples,
            "proposal_mean": proposal_mean,
            "energy_mean_est": energy_mean_est,
            "log_importance_weights": log_ratios,
        }

    @tf.function(experimental_relax_shapes=True)
    def _process_instance_autoregressive_batch(
        self,
        x,
        observed_mask,
        missing_mask,
        u_inds,
        num_importance_samples,
        seed=None,
    ):
        input_dict = make_input_dict(x, observed_mask, missing_mask)

        outputs = self.call(
            input_dict,
            num_importance_samples=num_importance_samples,
            seed=seed,
            selected_indices=u_inds[:, tf.newaxis],
        )

        return (
            tf.reduce_sum(outputs["energy_log_prob"]),
            tf.reduce_sum(outputs["proposal_log_prob"]),
        )

    def _instance_autoregressive_log_prob(
        self,
        x,
        observed_mask,
        missing_mask,
        num_importance_samples,
        num_permutations,
        seed=None,
    ):
        u_inds = np.where(((1.0 - observed_mask) * (1.0 - missing_mask)) == 1)[0]

        if len(u_inds) == 0:
            return 0.0, 0.0

        energy_results = []
        proposal_results = []

        for _ in range(num_permutations):
            u_inds = np.random.permutation(u_inds)
            curr_observed_mask = np.array(observed_mask)

            observed_mask_batch = []
            for i in u_inds:
                observed_mask_batch.append(curr_observed_mask.copy())
                curr_observed_mask[i] = 1

            x_batch = np.tile(x, [len(u_inds), 1])
            missing_mask_batch = np.tile(missing_mask, [len(u_inds), 1])
            observed_mask_batch = np.array(observed_mask_batch).astype(x_batch.dtype)

            (
                energy_log_prob,
                proposal_log_prob,
            ) = self._process_instance_autoregressive_batch(
                x_batch,
                observed_mask_batch,
                missing_mask_batch,
                u_inds,
                num_importance_samples,
                seed,
            )

            energy_results.append(energy_log_prob)
            proposal_results.append(proposal_log_prob)

        energy_mean_log_prob = tfp.math.reduce_logmeanexp(energy_results).numpy()
        proposal_mean_log_prob = tfp.math.reduce_logmeanexp(proposal_results).numpy()

        return energy_mean_log_prob, proposal_mean_log_prob

    @tf.function
    def _batch_non_autoregressive_log_prob(
        self, x, observed_mask, missing_mask, seed=None
    ):
        input_dict = make_input_dict(x, observed_mask, missing_mask)

        outputs = self.call(
            input_dict,
            num_importance_samples=1,
            seed=seed,
        )

        return outputs["energy_log_prob"], outputs["proposal_log_prob"]

    def log_prob(
        self,
        x: Union[np.ndarray, tf.Tensor],
        observed_mask: Union[np.ndarray, tf.Tensor],
        missing_mask: Optional[Union[np.ndarray, tf.Tensor]] = None,
        num_importance_samples: int = 20,
        num_permutations: int = 1,
        autoregressive: bool = True,
        non_autoregressive_batch_size: int = 32,
    ) -> Tuple[np.ndarray, np.ndarray]:
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
                of each instance to autoregressively compute log probabilities for. The
                returned log probability for each instance is the mean of the
                probabilities that were computed for each permutation.
            autoregressive: A boolean indicating if log probabilities should be
                computed autoregressively. If False, individual log probabilities for
                each unobserved dimension are returned.
            non_autoregressive_batch_size: The batch size to use when `autoregressive`
                is False.

        Returns:
            Two arrays with the energy log probabilities and the proposal log
            probabilities. If `autoregressive` is True, each array has shape
            `[batch_size]`. If `autoregressive` is False, each array has shape,
            `[batch_size, num_features]`.
        """
        results = []

        if missing_mask is None:
            missing_mask = np.zeros_like(observed_mask)

        if autoregressive:
            for batch in zip(x, observed_mask, missing_mask):
                results.append(
                    self._instance_autoregressive_log_prob(
                        *batch, num_importance_samples, num_permutations
                    )
                )
        else:
            data_iter = tf.data.Dataset.from_tensor_slices(
                (x, observed_mask, missing_mask)
            ).batch(non_autoregressive_batch_size)

            for batch in data_iter:
                results.append((self._batch_non_autoregressive_log_prob(*batch)))

        energy_log_probs, proposal_log_probs = zip(*results)

        return np.array(energy_log_probs), np.array(proposal_log_probs)

    @tf.function
    def _dim_proposal_sample(self, x_o, observed_mask, index, seed):
        proposal_outputs = self._proposal_network([x_o, observed_mask])
        proposal_dist, _ = self._create_proposal_dist(proposal_outputs)
        proposal_samples = proposal_dist.sample(1, seed)
        return proposal_samples[0, :, index]

    @tf.function
    def _dim_energy_sample(
        self, x_o, observed_mask, index, seed, num_resampling_samples
    ):
        input_dict = make_input_dict(x_o, observed_mask)
        outputs = self.call(
            input_dict,
            num_importance_samples=num_resampling_samples,
            seed=seed,
            selected_indices=tf.expand_dims(index, 0),
        )
        log_importance_weights = outputs["log_importance_weights"][..., 0]
        resampled_inds = tf.cast(
            tf.random.categorical(log_importance_weights, 1), tf.int32
        )
        winners = tf.gather(
            outputs["proposal_samples"][..., 0], resampled_inds, batch_dims=1
        )
        return winners[:, 0]

    def _instance_autoregressive_sample(
        self,
        x_o,
        observed_mask,
        num_samples,
        num_resampling_samples,
        use_proposal,
        seed=None,
    ):
        u_inds = np.where((1.0 - observed_mask) == 1)[0]
        u_inds = np.random.permutation(u_inds)

        x_o = np.copy(x_o)
        curr_x_o = np.tile(x_o[np.newaxis, ...], [num_samples, 1])
        curr_observed_mask = np.tile(observed_mask[np.newaxis, ...], [num_samples, 1])

        for j in u_inds:
            if use_proposal:
                dim_samples = self._dim_proposal_sample(
                    curr_x_o, curr_observed_mask, j, seed
                )
            else:
                dim_samples = self._dim_energy_sample(
                    curr_x_o, curr_observed_mask, j, seed, num_resampling_samples
                )

            curr_x_o[:, j] = dim_samples
            curr_observed_mask[:, j] = 1

        return curr_x_o

    def sample(
        self,
        x: Union[np.ndarray, tf.Tensor],
        observed_mask: Union[np.ndarray, tf.Tensor],
        num_samples: int = 1,
        num_resampling_samples: int = 100,
        use_proposal: bool = False,
    ) -> np.ndarray:
        """Samples the given conditional distribution.

        Args:
            x: A tensor of shape `[batch_size, num_features]` that contains the values
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
            `num_samples` is 1, in which case the array will have size
            `[batch_size, num_features]`.
        """
        results = []

        for batch in zip(x, observed_mask):
            results.append(
                self._instance_autoregressive_sample(
                    *batch,
                    num_samples,
                    num_resampling_samples,
                    use_proposal,
                )
            )

        samples = np.array(results)
        if num_samples == 1:
            samples = np.squeeze(samples, axis=1)

        return samples

    @tf.function
    def _get_batch_mean(self, x, observed_mask, num_importance_samples, seed=None):
        input_dict = make_input_dict(x, observed_mask)
        outputs = self.call(
            input_dict, num_importance_samples=num_importance_samples, seed=seed
        )
        return outputs["energy_mean_est"], outputs["proposal_mean"]

    def mean(
        self,
        x: Union[np.ndarray, tf.Tensor],
        observed_mask: Union[np.ndarray, tf.Tensor],
        num_importance_samples: int = 100,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Computes the means of the specified conditional distributions.

        Args:
            x: A tensor of shape `[batch_size, num_features]` that contains the values
                of the oberved and unobserved features.
            observed_mask: A binary tensor of shape `[batch_size, num_features]` that
                if 1 for features which are observed and 0 otherwise.
            num_importance_samples: The number of importance samples to use when
                estimating the mean of the energy distribution.

        Returns:
            Two arrays of shape `[batch_size, num_features]` that contains the means
            of the energy and proposal distributions.
        """
        energy_means, proposal_means = self._get_batch_mean(
            x,
            observed_mask,
            num_importance_samples,
        )

        energy_means = np.where(observed_mask, x, energy_means)
        proposal_means = np.where(observed_mask, x, proposal_means)

        return energy_means, proposal_means


def load_model(directory: str) -> Tuple[ACE, str]:
    """Loads a trained ACE model.

    Args:
        directory: The directory of the model that should be loaded. This is the
            directory that was created during training.

    Returns:
        A tuple with the loaded ACE model and the name of the dataset it was trained on.
    """
    config_path = os.path.join(directory, "full-config.gin")
    gin.parse_config_file(config_path, skip_unknown=True)

    model = ACE()

    dataset = None
    with open(config_path, "r") as fp:
        for line in fp:
            if line.startswith("train.dataset"):
                dataset = line.split(" = ")[1].strip("'\n")

    if dataset is None:
        raise ValueError("Could not find value for 'train.dataset' in config file.")

    checkpoint_dir = os.path.join(directory, "checkpoints")
    ckpt = tf.train.Checkpoint(model)
    checkpoint_name = tf.train.latest_checkpoint(checkpoint_dir)
    ckpt.restore(checkpoint_name).expect_partial()

    return model, dataset
