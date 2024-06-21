# metMHN

Mutual Hazard Networks for metastatic disease or in short [metMHN](https://www.biorxiv.org/content/10.1101/2024.01.30.577989v1) is an extension to the MHN-algorithm by [Schill et al. (2020)](https://academic.oup.com/bioinformatics/article/36/1/241/5524604) and [Schill et al. (2023)](https://www.biorxiv.org/content/10.1101/2023.12.03.569824v1) to account for the joint evolution of primary tumor and metastasis pairs. It accounts for sampling bias and different primary/tumor metastasis diagnosis orderings. 

## Installation
We advise to use a virtual environment.
Create a new virtual environment

```bash
python3 -m venv .venv
```

Activate the virtual environment

```bash
source .venv/bin/activate
```
We rely on the [JAX](https://github.com/google/jax) library for our computations. If you **don't have access to a gpu** please install the cpu-only version of the libraries by running: 

```bash
pip3 install -r requirements.txt
```
If you have access to a gpu, first install the requirements as detailed above and then install the jaxlib cuda package from [here](https://jax.readthedocs.io/en/latest/installation.html#pip-installation-gpu-cuda-installed-via-pip-easier) 


Finally install the metMHN package locally using

```bash
pip install -e .
```
## Run metMHN
We provide an example analysis of a reduced lung adeno carcinoma (LUAD) dataset that can be run on a standard desktop by first starting a jupyter notebook server
```bash
jupyter notebook
```
and then by executing the file `examples/data_analysis.ipynb`.\
Additionally we provide an example of simple simulation based post-hoc analyses in `examples/simulation_study.ipynb`.\
Alternatively metMHN can also be run from the command line by
```bash
python3 examples/analysis.py -args <input-annotation-file.csv> <input-event-data.csv> <output-inferred-model.csv>
```
where `<input-annotation-file.csv>` contains supporting information for each patient, `<input-event-data.csv>` contains the binarized primary tumor/metastasis genotypes for each patient and `<output-inferred-model.csv>` is the file where the final inferred metMHN should be stored.

You can also set the following command line arguments:
|Argument | Description|
| --- | ---|
|-cv | Boolean, If set, perform crossvalidation|
|-cv_start | Float, Lower limit of hyperparameter range to test in crossvalidation, defaults to 1e-05 |
|-cv_end | Float, Upper limit of hyperparameter range to test in crossvalidation, defaults to 1e-02|
|-cv_fold | Integer, Number of crossvalidation folds, defaults to 5|
|-cv_splits | Integer, Number of hyperparameters to test in the range cv_start to cv_end, defaults to 5 |
|-pm_ratio| Float, Expected ratio of never metastasizing primary tumors to metastasizing primary tumors|
|-lam | Float, Weight of penalization. Should only be set if no cross validation is performed|
|-logs| String, relative filepath for log-files|
|-seed| Integer, Seed to be used random number generator|

We also provide a script to assess how well metMHN can recover groundtruth parameters:
```bash
python3 examples/recall_study.py -args 
```
You can also set the following command line arguments:
|Argument | Description|
| --- | ---|
|-n_dat | Integer, Number of datapoints in simulated dataset|
|-n_reps | Integer, Number of simulated datasets to generate|
|-logs | String, Relative filepath for log-files|
|-seed | Integer, Seed for random number generator|

