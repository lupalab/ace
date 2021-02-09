[1]: https://arxiv.org/abs/2102.04426

# Arbitrary Conditioning with Energy

This is the official implementation of the paper ["Arbitrary Conditional Distributions
with Energy"][1].

Arbitrary Conditioning with Energy (ACE) is a method that can simultaneously estimate
the distribution `p(x_u | x_o)` for all possible subsets of features `x_u` and `x_o`. 
ACE achieves state-of-the-art for arbitrary conditional and marginal likelihood
estimation and for tabular data imputation.

## Installation

The `ace` package can be installed by cloning this repository then running
```
pip install .
```
in the directory of this README file. This should ensure all necessary dependencies
are installed.

### UCI Datasets

Once the package is installed, the UCI datasets that are used in the paper can be
installed with:
```
python -m ace.data.download
```
This will download and preprocess the UCI datasets so that they can be used for
training models.

### Custom Datasets

ACE models can easily be trained on your own custom datasets. In order to do this,
the data must be provided in a specific format. Your data should be placed in its own
directory that contains exactly three files in one of two formats:
`train.csv`, `val.csv`, `test.csv`, or `train.npy`, `val.npy`, `test.npy`. These
three files will contain the training data, validation data, and test data. If using
the CSV format, the file should contain no headers. It is okay if the data contains
missing values. In NumPy format, the missing values should be filled with `NaN`.
To use your custom dataset, you will provide the path to the directory that contains
your data as the `train.dataset` argument in the configuration file (discussed below).

## Usage

### Training

Models can be trained using the command-line interface accessed with:
```
python -m ace.train --help
```
We use [Gin](https://github.com/google/gin-config) to configure model and training
settings. Example configuration files are provided in [`config/`](config). For
example, you can train a model on the Gas dataset with:
```
python -m ace.train --config config/gas.gin
```
Model weights and TensorBoard summaries will be saved within the directory specified by
`train.log_dir` in the configuration file. Note that the training script will append a
timestamp to whatever directory you specify.

The `train_ace` method (see [`train.py`](ace/train.py)) and `ACE` class
(see [`ace.py`](ace/ace.py)) are the two things registered with Gin, and any of their
arguments can be specified in a custom configuration file. Refer to their documentation
for the meanings of the various arguments.

### Evaluation

The interface for evaluating models is accessed with:
```
python -m ace.evaluate --help
```
This interface allows you to perform the same evaluations used in the paper.

For example, you could evaluate a model's imputation performance on the test data with:
```
python -m ace.evaluate --model_dir path/to/model/dir/ --run imputation --num_importance_samples 20000 --use_test_set
```
The path provided to the `--model_dir` option should be the directory that was created
by the training script to house the model (e.g., `logs/UCI/gas/2021-01-04-15-27-53`).

Evaluation results will be saved within an `evaluations/` directory inside the model's
directory. The pertinent metrics will be in files called `results.json`.

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