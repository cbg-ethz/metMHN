# metMHN

MHNs for primary tumors and metastases

## Setting up the Python environment

Create a new virtual environment

```bash
python3 -m venv .venv
```

Activate the virtual environment

```bash
source .venv/bin/activate
```

Install the required packages

```bash
pip3 install -r requirements.txt
```

Then install jaxlib cuda package from [here](https://jax.readthedocs.io/en/latest/installation.html#pip-installation-gpu-cuda-installed-via-pip-easier) choosing the installation where CUDA is installed via pip and deciding between CUDA 11 and CUDA 12.
Finally install the metMHN package locally using

```bash
pip install -e .
```
