# MetaMHN
This is a adaptation to MHN so that it works on metastasis cancer data

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
Then install jaxlib cuda package from [here](https://github.com/google/jax#pip-installation-gpu-cuda-installed-via-pip-easier) choosing the installation where CUDA is installed via pip.