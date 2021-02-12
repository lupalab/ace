import setuptools

with open("README.md", "r") as fp:
    long_description = fp.read()

setuptools.setup(
    name="ace",
    version="1.0",
    packages=setuptools.find_packages(),
    python_requires=">=3.7",
    long_description=long_description,
    long_description_content_type="text/markdown",
    description='Official code for "Arbitrary Conditional Distributions with Energy".',
    url="https://github.com/lupalab/ace",
    author="Ryan Strauss",
    author_email="rrs@cs.unc.edu",
    install_requires=[
        "numpy",
        "tensorflow==2.4",
        "tensorflow-probability==0.11",
        "tqdm",
        "click",
        "pandas",
        "h5py",
        "gin-config",
    ],
)
