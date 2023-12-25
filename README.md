# metMHN

metMHN is an extension to the MHN-algorithm by Schill et al. (2019) and Schill et al. (2023) to account for the joint evolution of primary tumor and metastasis pairs. It accounts for sampling bias and different primary/tumor metastasis diagnosis orders (i.e. synchroneous vs. metachronous diagnosis)

## Installation
we advise to use a virtual environment.
Create a new virtual environment

```bash
python3 -m venv .venv
```

Activate the virtual environment

```bash
source .venv/bin/activate
```
We rely on the [JAX] (https://github.com/google/jax) library for our computations. If you **don't have access to a gpu** please install the cpu-only version of the libraries by running: 

```bash
pip3 install -r requirements.txt
```
If you have access to a gpu, first install the requirements as detailed above and then install the jaxlib cuda package from [here](https://jax.readthedocs.io/en/latest/installation.html#pip-installation-gpu-cuda-installed-via-pip-easier) 


Finally install the metMHN package locally using

```bash
pip install -e .
```
