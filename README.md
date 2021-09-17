[1]: https://arxiv.org/abs/2102.04426

# Arbitrary Conditioning with Energy

This is the official repository for the paper ["Arbitrary Conditional Distributions
with Energy"][1].

Arbitrary Conditioning with Energy (ACE) is a method that can simultaneously estimate
the distribution `p(x_u | x_o)` for all possible subsets of features `x_u` and `x_o`. 
ACE achieves state-of-the-art for arbitrary conditional and marginal likelihood
estimation and for tabular data imputation.

In this repository, we provide a simple and accessible implementation of the most
straightforward version of ACE. The model is implemented in TensorFlow as a
`tf.keras.Model` in order to facilitate easy custom use.

## Installation

The `ace` package can be installed by cloning this repository then running
```
pip install .
```
in the directory of this README file. This should ensure all necessary dependencies
are installed.

### UCI Datasets

The UCI datasets that are used in the paper can be installed as 
[TensorFlow Datasets](https://www.tensorflow.org/datasets). To install one of the
datasets, navigate to that dataset's directory (e.g. [`datasets/gas`](datasets/gas)),
the run:
```
tfds build
```
This will download and prepare the dataset so that it can be accessed via
TensorFlow Datasets when the training script is run.

## Usage

### Training

Models can be trained using the command-line interface accessed with:
```
python -m ace.train --help
```
We use [Gin](https://github.com/google/gin-config) to configure model and training
settings. Example configuration files are provided in [`configs/`](configs). For
example, you can train a model on the Gas dataset with:
```
python -m ace.train --config config/gas.gin --logdir logs/gas
```
Model weights and TensorBoard summaries will be saved within the directory specified by
the `--logdir` option.

### Evaluation

An interface for simple evaluations of models on test data is accessed with:
```
python -m ace.test --help
```

For example, you could evaluate a model's imputation performance with:
```
python -m ace.test --model_dir path/to/model/dir/ --run imputation --num_importance_samples 20000
```
The path provided to the `--model_dir` option should be the directory that was created
by the training script to house the model.

Evaluation results will be saved within an `evaluations/` directory inside the model's
directory. The pertinent metrics will be in files called `results.json`.

### Custom Use

Using an ACE model in your own code can be easily accomplished as follows:
```python
from ace import ACEModel

# When creating an ACE model, the dimensionality of the data must be provided (e.g., 8)
model = ACEModel(num_features=8)
```

This class implements the standard version of ACE that is for use with continuous data.
However, it can be adapted to work with discrete data with some small modification, as
described in the paper.

The model is called with `x` (the data) and `b` (the binary mask indicating which
features are observed) as mandatory inputs. A `missing_mask` can optionally be provided
to indicate which features are missing (i.e., will be neither observed nor unobserved).
An `ACEOutputs` object will be returned, which contains various outputs of the model.

```python
# Use your real data here in practice.
x = np.random.normal(size=(32, 8))
b = np.random.choice(2, size=(32, 8))

outputs = model(
    [x, b],
    missing_mask=None,
    num_importance_samples=10,
    training=False,
)
```

The class also provides methods for computing autoregressive likelihoods, sampling, and
imputing.

```python
energy_log_probs, proposal_log_probs = model.log_prob(x, b)
samples = model.sample(x, b)
energy_imputations, proposal_imputations = model.impute(x, b)
```

The model can be trained with `model.fit()` (see [`train.py`](ace/train.py) for an
example).

In order to train with data that has missing values, the `tf.data.Dataset` that is
provided to the `fit` method should have three tensors instead of just two (the third
being the binary mask to indicate which values are missing).

## Citation

```
@misc{strauss2021arbitrary,
      title={Arbitrary Conditional Distributions with Energy}, 
      author={Ryan R. Strauss and Junier B. Oliva},
      year={2021},
      eprint={2102.04426},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
