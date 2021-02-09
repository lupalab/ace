import json
import os
from datetime import datetime
from typing import Tuple

import click
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from ace.ace import load_model, ACE
from ace.data import load_data, get_mask_generator
from ace.data.masking import MaskGenerator
from ace.utils import enable_gpu_growth, nrmse_score

mixed_precision = tf.keras.mixed_precision


def _add_observed_mask_fn(mask_generator):
    def fn(t):
        [mask] = tf.py_function(mask_generator, [t.get_shape().as_list()], [tf.float32])
        return t, mask

    return fn


def _remove_masks(x, *args):
    return x


def _add_marginal_masks_fn(marginal_dims):
    def fn(x):
        missing = tf.greater_equal(
            tf.range(tf.shape(x)[-1]),
            marginal_dims,
        )
        missing = tf.cast(missing, x.dtype)
        return x, tf.zeros_like(x), missing

    return fn


def _save_likelihoods_json(
    dir_path, filename, energy_likelihoods, proposal_likelihoods
):
    energy_likelihoods = np.mean(energy_likelihoods, axis=-1)
    energy_mean = np.mean(energy_likelihoods)
    energy_std = np.std(energy_likelihoods)

    proposal_likelihoods = np.mean(proposal_likelihoods, axis=-1)
    proposal_mean = np.mean(proposal_likelihoods)
    proposal_std = np.std(proposal_likelihoods)

    data = {
        "energy_mean": float(energy_mean),
        "energy_std": float(energy_std),
        "proposal_mean": float(proposal_mean),
        "proposal_std": float(proposal_std),
    }

    with open(os.path.join(dir_path, filename), "w") as fp:
        json.dump(data, fp)

    return data


def _evaluate_likelihoods(
    model, dataset, num_trials, num_importance_samples, num_permutations
):
    energy_results = []
    proposal_results = []

    for i in range(num_trials):
        log_probs = [
            model.log_prob(
                *batch,
                num_importance_samples=num_importance_samples,
                num_permutations=num_permutations,
            )
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
    model: ACE,
    dataset: tf.data.Dataset,
    mask_generator: MaskGenerator,
    num_trials: int,
    num_permutations: int,
    num_importance_samples: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if isinstance(dataset.element_spec, tuple):
        dataset = dataset.map(_remove_masks)

    add_observed_mask = _add_observed_mask_fn(mask_generator)
    dataset = dataset.map(add_observed_mask)
    dataset = dataset.batch(1)

    return _evaluate_likelihoods(
        model, dataset, num_trials, num_importance_samples, num_permutations
    )


def evaluate_marginal_likelihoods(
    model: ACE,
    dataset: tf.data.Dataset,
    num_trials: int,
    num_permutations: int,
    num_importance_samples: int,
    marginal_dims: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if isinstance(dataset.element_spec, tuple):
        dataset = dataset.map(_remove_masks)

    if marginal_dims == -1:
        marginal_dims = dataset.element_spec.shape[-1]

    add_marginal_masks = _add_marginal_masks_fn(marginal_dims)
    dataset = dataset.map(add_marginal_masks)
    dataset = dataset.batch(1)

    return _evaluate_likelihoods(
        model, dataset, num_trials, num_importance_samples, num_permutations
    )


def evaluate_imputation(
    model: ACE,
    dataset: tf.data.Dataset,
    mask_generator: MaskGenerator,
    num_trials: int,
    num_importance_samples: int,
    batch_size: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not isinstance(dataset.element_spec, tuple):
        add_observed_mask = _add_observed_mask_fn(mask_generator)
        dataset = dataset.map(add_observed_mask)
    else:

        def flip_mask(x, missing_mask):
            return x, 1 - missing_mask

        dataset = dataset.map(flip_mask)

    dataset = dataset.batch(batch_size)

    energy_imputations = []
    proposal_imputations = []
    x_batches = []
    observed_mask_batches = []

    for i in range(num_trials):
        energy_imputations.append([])
        proposal_imputations.append([])
        x_batches.append([])
        observed_mask_batches.append([])

        for x, observed_mask in tqdm(
            dataset, desc=f"Sampling (Trial {i + 1}/{num_trials})"
        ):
            batch_energy_imputations, batch_proposal_imputations = model.mean(
                x, observed_mask, num_importance_samples
            )

            energy_imputations[-1].append(batch_energy_imputations)
            proposal_imputations[-1].append(batch_proposal_imputations)
            x_batches[-1].append(x)
            observed_mask_batches[-1].append(observed_mask)

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


@click.command(name="evaluate")
@click.option(
    "--model_dir",
    type=click.Path(exists=True, file_okay=False),
    nargs=1,
    required=True,
    help="Path of the directory of the model to evaluate.",
)
@click.option(
    "--run",
    type=click.Choice(
        ["ac-likelihoods", "marginal-likelihoods", "joint-likelihoods", "imputation"]
    ),
    multiple=True,
    required=True,
    help="The type of evaluation to run. This option can be provided multiple "
    "times to run multiple evaluations.",
)
@click.option(
    "--eager",
    is_flag=True,
    help="If flag is set, eager execution will be enabled inside tf.functions.",
)
@click.option(
    "--growth/--no-growth",
    default=True,
    help="Whether or not GPU growth will be enabled.",
)
@click.option(
    "--use_mixed_precision",
    is_flag=True,
    help="If flag is set, mixed precision mode will be enabled.",
)
@click.option(
    "--use_subset_val",
    is_flag=True,
    help="If flag is set, at most 2048 samples will be used for validation.",
)
@click.option(
    "--use_test_set",
    is_flag=True,
    help="Whether or not to use the test set (as opposed to the validation set).",
)
@click.option(
    "--num_trials",
    type=click.INT,
    nargs=1,
    default=1,
    help="Number of times to perform evaluation. For computing standard deviations.",
)
@click.option(
    "--num_importance_samples",
    type=click.INT,
    nargs=1,
    default=100,
    help="Number of importance samples to use.",
)
@click.option(
    "--num_permutations",
    type=click.INT,
    nargs=1,
    default=1,
    help="Number of permutations of unobserved indices to average over when computing "
    "likelihoods for a given instance.",
)
@click.option(
    "--batch_size",
    type=click.INT,
    nargs=1,
    default=32,
    help="Batch size that is used when generating imputations.",
)
def _main(
    model_dir,
    run,
    eager,
    growth,
    use_mixed_precision,
    use_subset_val,
    use_test_set,
    num_trials,
    num_importance_samples,
    num_permutations,
    batch_size,
):
    """Evaluates a trained ACE model."""
    tf.config.run_functions_eagerly(eager)

    if growth:
        enable_gpu_growth()

    if use_mixed_precision:
        policy = mixed_precision.Policy("mixed_float16")
        mixed_precision.set_global_policy(policy)

    model, dataset = load_model(model_dir)

    _, val_dataset, test_dataset = load_data(dataset, use_subset_val)
    eval_dataset = test_dataset if use_test_set else val_dataset
    mask_generator = get_mask_generator("bernoulli")

    eval_dir = os.path.join(
        model_dir, "evaluations", datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    )
    os.makedirs(eval_dir)

    with open(os.path.join(eval_dir, "params.json"), "w") as fp:
        json.dump(
            {
                "run": run,
                "use_test_set": use_test_set,
                "use_subset_val": use_subset_val,
                "num_trials": num_trials,
                "num_importance_samples": num_importance_samples,
                "num_permutations": num_permutations,
            },
            fp,
        )

    if "ac-likelihoods" in run:
        ac_dir = os.path.join(eval_dir, "ac-likelihoods")
        os.makedirs(ac_dir)

        energy_likelihoods, proposal_likelihoods = evaluate_ac_likelihoods(
            model,
            eval_dataset,
            mask_generator,
            num_trials,
            num_permutations,
            num_importance_samples,
        )

        np.save(os.path.join(ac_dir, "energy-likelihoods.npy"), energy_likelihoods)
        np.save(os.path.join(ac_dir, "proposal-likelihoods.npy"), proposal_likelihoods)

        data = _save_likelihoods_json(
            ac_dir, "results.json", energy_likelihoods, proposal_likelihoods
        )

        print("------------------------")
        print("AC Likelihoods")
        print("------------------------")
        print("Energy:   {:.3f}".format(data["energy_mean"]))
        print("Proposal: {:.3f}".format(data["proposal_mean"]))
        print("------------------------\n")

    if "marginal-likelihoods" in run:
        marginal_dir = os.path.join(eval_dir, "marginal-likelihoods")
        os.makedirs(marginal_dir)

        for dims in [3, 5, 10]:
            energy_likelihoods, proposal_likelihoods = evaluate_marginal_likelihoods(
                model,
                eval_dataset,
                num_trials,
                num_permutations,
                num_importance_samples,
                dims,
            )

            np.save(
                os.path.join(marginal_dir, f"energy-likelihoods-{dims}-dims.npy"),
                energy_likelihoods,
            )
            np.save(
                os.path.join(marginal_dir, f"proposal-likelihoods-{dims}-dims.npy"),
                proposal_likelihoods,
            )

            data = _save_likelihoods_json(
                marginal_dir,
                f"results-{dims}-dims.json",
                energy_likelihoods,
                proposal_likelihoods,
            )

            print("------------------------")
            print("Marginal-{} Likelihoods".format(dims))
            print("------------------------")
            print("Energy:   {:.3f}".format(data["energy_mean"]))
            print("Proposal: {:.3f}".format(data["proposal_mean"]))
            print("------------------------\n")

    if "joint-likelihoods" in run:
        joint_dir = os.path.join(eval_dir, "joint-likelihoods")
        os.makedirs(joint_dir)

        energy_likelihoods, proposal_likelihoods = evaluate_marginal_likelihoods(
            model,
            eval_dataset,
            num_trials,
            num_permutations,
            num_importance_samples,
            marginal_dims=-1,
        )

        np.save(os.path.join(joint_dir, "energy-likelihoods.npy"), energy_likelihoods)
        np.save(
            os.path.join(joint_dir, "proposal-likelihoods.npy"), proposal_likelihoods
        )

        data = _save_likelihoods_json(
            joint_dir, "results.json", energy_likelihoods, proposal_likelihoods
        )

        print("------------------------")
        print("Joint Likelihoods")
        print("------------------------")
        print("Energy:   {:.3f}".format(data["energy_mean"]))
        print("Proposal: {:.3f}".format(data["proposal_mean"]))
        print("------------------------\n")

    if "imputation" in run:
        imputation_dir = os.path.join(eval_dir, "imputation")
        os.makedirs(imputation_dir)

        (
            energy_nrmse,
            proposal_nrmse,
            energy_imputations,
            proposal_imputations,
        ) = evaluate_imputation(
            model,
            eval_dataset,
            mask_generator,
            num_trials,
            num_importance_samples,
            batch_size,
        )

        np.save(
            os.path.join(imputation_dir, "energy-imputations.npy"), energy_imputations
        )
        np.save(
            os.path.join(imputation_dir, "proposal-imputations.npy"),
            proposal_imputations,
        )
        np.save(os.path.join(imputation_dir, "energy-nrmse.npy"), energy_nrmse)
        np.save(os.path.join(imputation_dir, "proposal-nrmse.npy"), proposal_nrmse)

        data = {
            "energy_nrmse_mean": float(np.mean(energy_nrmse)),
            "energy_nrmse_std": float(np.std(energy_nrmse)),
            "proposal_nrmse_mean": float(np.mean(proposal_nrmse)),
            "proposal_nrmse_std": float(np.std(proposal_nrmse)),
        }

        with open(os.path.join(imputation_dir, "results.json"), "w") as fp:
            json.dump(data, fp)

            print("------------------------")
            print("Imputation NRMSE")
            print("------------------------")
            print("Energy:   {:.3f}".format(data["energy_nrmse_mean"]))
            print("Proposal: {:.3f}".format(data["proposal_nrmse_mean"]))
            print("------------------------")


if __name__ == "__main__":
    _main()
