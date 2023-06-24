from typing import Tuple

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from ace import ACEModel
from ace.masking import MaskGenerator, get_add_mask_fn, get_add_marginal_masks_fn


def nrmse_score(
    imputations: np.ndarray, true_data: np.ndarray, observed_mask: np.ndarray
) -> np.ndarray:
    """Calculates the normalized root-mean-square error metric for imputed values.

    Args:
        imputations: A numpy array of shape `[..., num_instances, num_features]` that
            contains the imputed data.
        true_data: A numpy array of shape `[..., num_instances, num_features]` that
            contains the ground truth data.
        observed_mask: A numpy array of shape `[..., num_instances, num_features]` that
            contains the binary mask indicating which features are observed.

    Returns:
        A numpy array with shape `imputations.shape[:-2]` that contains the
        NRMSE scores.
    """
    error = (imputations - true_data) ** 2
    mse = np.sum(error, axis=-2) / np.count_nonzero(1.0 - observed_mask, axis=-2)
    nrmse = np.sqrt(mse) / np.std(true_data, axis=-2)
    print(nrmse)
    return np.nanmean(nrmse, axis=-1)


def _evaluate_likelihoods(
    model, dataset, num_trials, num_importance_samples, num_permutations
):
    energy_results = []
    proposal_results = []

    @tf.function
    def get_lls(*args):
        return model.log_prob(
            *args,
            num_importance_samples=num_importance_samples,
            num_permutations=num_permutations,
        )

    for i in range(num_trials):
        log_probs = [
            get_lls(*batch)
            for batch in tqdm(
                dataset, desc=f"Computing Likelihoods (Trial {i + 1}/{num_trials})"
            )
        ]

        energy_logprobs, proposal_logprobs = zip(*log_probs)
        energy_results.append(np.vstack(energy_logprobs))
        proposal_results.append(np.vstack(proposal_logprobs))

    energy_results = np.squeeze(energy_results, axis=-1)
    proposal_results = np.squeeze(proposal_results, axis=-1)

    return energy_results, proposal_results


def evaluate_ac_likelihoods(
    model: ACEModel,
    dataset: tf.data.Dataset,
    mask_generator: MaskGenerator,
    num_trials: int = 1,
    num_permutations: int = 1,
    num_importance_samples: int = 100,
) -> Tuple[np.ndarray, np.ndarray]:
    add_mask_fn = get_add_mask_fn(mask_generator)
    dataset = dataset.map(add_mask_fn)

    return _evaluate_likelihoods(
        model, dataset, num_trials, num_importance_samples, num_permutations
    )


def evaluate_marginal_likelihoods(
    model: ACEModel,
    dataset: tf.data.Dataset,
    marginal_dims: int,
    num_trials: int = 1,
    num_permutations: int = 1,
    num_importance_samples: int = 100,
) -> Tuple[np.ndarray, np.ndarray]:
    if marginal_dims == -1:
        marginal_dims = dataset.element_spec.shape[-1]

    add_marginal_masks = get_add_marginal_masks_fn(marginal_dims)
    dataset = dataset.map(add_marginal_masks)

    return _evaluate_likelihoods(
        model, dataset, num_trials, num_importance_samples, num_permutations
    )


def evaluate_imputation(
    model: ACEModel,
    dataset: tf.data.Dataset,
    mask_generator: MaskGenerator,
    mask_fn = None,
    num_trials: int = 1,
    num_importance_samples: int = 100,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    add_mask_fn = get_add_mask_fn(mask_generator) if mask_fn is None else get_add_mask_fn(mask_fn)
    dataset = dataset.map(add_mask_fn)

    energy_imputations = []
    proposal_imputations = []
    x_batches = []
    observed_mask_batches = []

    @tf.function
    def get_imputations(x, b):
        return model.impute(x, b, num_importance_samples)

    for i in range(num_trials):
        energy_imputations.append([])
        proposal_imputations.append([])
        x_batches.append([])
        observed_mask_batches.append([])

        for x, b in tqdm(dataset, desc=f"Sampling (Trial {i + 1}/{num_trials})"):
            batch_energy_imputations, batch_proposal_imputations = get_imputations(x, b)

            energy_imputations[-1].append(batch_energy_imputations)
            proposal_imputations[-1].append(batch_proposal_imputations)
            x_batches[-1].append(x)
            observed_mask_batches[-1].append(b)

        energy_imputations[-1] = np.vstack(energy_imputations[-1])
        proposal_imputations[-1] = np.vstack(proposal_imputations[-1])
        x_batches[-1] = np.vstack(x_batches[-1])
        observed_mask_batches[-1] = np.vstack(observed_mask_batches[-1])

    energy_imputations = np.array(energy_imputations)
    proposal_imputations = np.array(proposal_imputations)
    x = np.array(x_batches)
    observed_mask = np.array(observed_mask_batches)

    energy_nrmse = nrmse_score(energy_imputations, x, observed_mask)
    proposal_nrmse = nrmse_score(proposal_imputations, x, observed_mask)

    return energy_nrmse, proposal_nrmse, energy_imputations, proposal_imputations
