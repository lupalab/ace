import json
import os

import click
import numpy as np
import tensorflow_datasets as tfds

from ace.evaluation import (
    evaluate_ac_likelihoods,
    evaluate_marginal_likelihoods,
    evaluate_imputation,
)
from ace.masking import BernoulliMaskGenerator, FixedMaskGenerator
from ace.utils import load_model, get_config_dict


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


@click.command()
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
        [
            "ac-likelihoods",
            "marginal-likelihoods",
            "joint-likelihoods",
            "imputation",
        ]
    ),
    multiple=True,
    required=True,
    help="The type of evaluation to run. This option can be provided multiple "
    "times to run multiple evaluations.",
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
    "--batch_size",
    type=click.INT,
    nargs=1,
    default=32,
    help="Batch size that is used when generating imputations.",
)
@click.option(
    "--num_instances",
    type=click.INT,
    nargs=1,
    help="Number of instances to evaluate.",
)
def test(model_dir, run, num_trials, num_importance_samples, batch_size, num_instances):
    model = load_model(model_dir)

    dataset = get_config_dict(model_dir)["train"]["dataset"]
    dataset = tfds.load(dataset)["test"].map(lambda x: x["features"])

    if num_instances is not None:
        dataset = dataset.take(num_instances)

    dataset = dataset.batch(batch_size)

    mask_generator = FixedMaskGenerator([[0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0]])

    if "ac-likelihoods" in run:
        eval_dir = os.path.join(model_dir, "evaluations", "ac-likelihoods")
        os.makedirs(eval_dir)

        energy_lls, proposal_lls = evaluate_ac_likelihoods(
            model,
            dataset,
            mask_generator,
            num_trials=num_trials,
            num_importance_samples=num_importance_samples,
        )

        np.save(os.path.join(eval_dir, "energy_lls.npy"), energy_lls)
        np.save(os.path.join(eval_dir, "proposal_lls.npy"), proposal_lls)

        data = _save_likelihoods_json(
            eval_dir, "results.json", energy_lls, proposal_lls
        )

        print("------------------------")
        print("AC Likelihoods")
        print("------------------------")
        print("Energy:   {:.3f}".format(data["energy_mean"]))
        print("Proposal: {:.3f}".format(data["proposal_mean"]))
        print("------------------------\n")

    if "marginal-likelihoods" in run:
        eval_dir = os.path.join(model_dir, "evaluations", "marginal-likelihoods")
        os.makedirs(eval_dir)

        for dims in [3, 5, 10]:
            energy_lls, proposal_lls = evaluate_marginal_likelihoods(
                model,
                dataset,
                dims,
                num_trials=num_trials,
                num_importance_samples=num_importance_samples,
            )

            np.save(
                os.path.join(eval_dir, f"energy_lls_{dims}_dims.npy"),
                energy_lls,
            )
            np.save(
                os.path.join(eval_dir, f"proposal_lls_{dims}_dims.npy"),
                proposal_lls,
            )

            data = _save_likelihoods_json(
                eval_dir,
                f"results-{dims}-dims.json",
                energy_lls,
                proposal_lls,
            )

            print("------------------------")
            print("Marginal-{} Likelihoods".format(dims))
            print("------------------------")
            print("Energy:   {:.3f}".format(data["energy_mean"]))
            print("Proposal: {:.3f}".format(data["proposal_mean"]))
            print("------------------------\n")

    if "imputation" in run:
        eval_dir = os.path.join(model_dir, "evaluations", "imputation")
        os.makedirs(eval_dir)

        (
            energy_nrmse,
            proposal_nrmse,
            energy_imputations,
            proposal_imputations,
        ) = evaluate_imputation(
            model,
            dataset,
            mask_generator,
            num_trials=num_trials,
            num_importance_samples=num_importance_samples,
        )

        np.save(os.path.join(eval_dir, "energy_imputations.npy"), energy_imputations)
        np.save(
            os.path.join(eval_dir, "proposal_imputations.npy"),
            proposal_imputations,
        )
        np.save(os.path.join(eval_dir, "energy_nrmse.npy"), energy_nrmse)
        np.save(os.path.join(eval_dir, "proposal_nrmse.npy"), proposal_nrmse)

        data = {
            "energy_nrmse_mean": float(np.mean(energy_nrmse)),
            "energy_nrmse_std": float(np.std(energy_nrmse)),
            "proposal_nrmse_mean": float(np.mean(proposal_nrmse)),
            "proposal_nrmse_std": float(np.std(proposal_nrmse)),
        }

        with open(os.path.join(eval_dir, "results.json"), "w") as fp:
            json.dump(data, fp)

            print("------------------------")
            print("Imputation NRMSE")
            print("------------------------")
            print("Energy:   {:.3f}".format(data["energy_nrmse_mean"]))
            print("Proposal: {:.3f}".format(data["proposal_nrmse_mean"]))
            print("------------------------")


if __name__ == "__main__":
    test()
