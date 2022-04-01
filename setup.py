import setuptools

with open("README.md", "r") as fp:
    long_description = fp.read()

setuptools.setup(
    name="ace",
    version="2.0",
    packages=setuptools.find_packages(),
    python_requires=">=3.8",
    long_description=long_description,
    long_description_content_type="text/markdown",
    description='Official code for the paper "Arbitrary Conditional Distributions with Energy".',
    url="https://github.com/lupalab/ace",
    author="Ryan Strauss",
    author_email="rrs@cs.unc.edu",
    install_requires=[
        "tensorflow>=2.5",
        "tensorflow-datasets",
        "tensorflow-probability",
        "numpy",
        "click",
        "gin-config",
        "loguru",
        "tqdm",
        "gdown",
    ],
)
