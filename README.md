<p align="center">
    <a style="text-decoration:none !important;" href="https://docs.python.org/3.12/" alt="Python3.12"> <img src="https://img.shields.io/badge/python-3.12-blue.svg" /> </a>
    <a style="text-decoration:none !important;" href="link to paper" alt="arXiv"> <img src="https://img.shields.io/badge/paper-AIIM-red" /> </a>
    <a style="text-decoration:none !important;" href="https://docs.conda.io/en/latest/miniconda.html" alt="package management"> <img src="https://img.shields.io/badge/conda-env-green" /> </a>
    <a style="text-decoration:none !important;" href="https://opensource.org/licenses/MIT" alt="License"> <img src="https://img.shields.io/badge/license-MIT-purple.svg" /> </a>
</p>

# Guideline-informed reinforcement learning for mechanical ventilation in critical care


## Paper
[Guideline informed reinforcement learning for mechanical ventilation in crticial care](https://link-to-paper.org)
```
@article{den2022guideline,
  title={Guideline-informed reinforcement learning for mechanical ventilation in critical care},
  author={den Hengst, F. and Otten, M. and Elbers, P. and Fran{\c{c}}ois-Lavet, V. and Hoogendoorn, M. and van Harmelen, F.},
  journal={Artificial Intelligence In Medicine},
  year={2023},
}
```

## Overview
This repository contains only code associated with the paper. The data needs to be accessed separately. For access to the data, we refer to the MIMIC [website](https://mimic-mit-edu.vu-nl.idm.oclc.org/docs/gettingstarted/).

To reproduce the results, follow these high-level steps which are described in more detail below:
1. getting started
2. patient data selection
3. clinically informed pre-processing
4. data-driven pre-processing and policy training
5. off-policy evaluation
6. analysis and evaluation

In case you run into issues, have a look at the FAQ below.


## 1. Getting started
To get started, clone this repository and run ``git submodule update``. To access the data, see the [MIMIC](https://mimic-mit-edu.vu-nl.idm.oclc.org/docs/gettingstarted/) website. You will need to take all the steps to acquire access PhysioNet. 

Once you have your credentions, use them to install MIMIC-III locally. This code base assumes you use Postgres and has only been tested on Postgres v14.9. Download [MIMIC-III v1.4](https://physionet.org/content/mimiciii/1.4/) locally and place the ``csv.gz`` files in the ``mimic-data`` directory. Next, build the database following the steps described in the [MIMIC guide](https://mimic-mit-edu.vu-nl.idm.oclc.org/docs/gettingstarted/local/install-mimic-locally-ubuntu/) for local installs. We here restate the main steps for Unix/Mac:
```bash
export repo_root=$(pwd)
export datadir="$repo_root/mimic-data"
export DBUSER=mimicuser
export DBNAME=mimic
export DBSCHEMA=mimiciii

# create an OS user for accessing mimic
createuser -P -sed $DBUSER
# connect to the database
psql -U $DBUSER -d postgres
```

Now create a new database and schema to hold the data. Replace the name of the database, owner and schema if you have chosen different values in the previous step.
```psql
CREATE DATABASE mimic OWNER mimicuser;
CREATE SCHEMA mimiciii;
```

```bash
export repo_root=$(pwd)
export datadir="$repo_root/mimic_data"
cd mimic-code/mimic-iii/buildmimic/postgres/
make mimic-gz $datadir DBUSER=$DBUSER DBNAME=$DBNAME DBSCHEMA=$DBSCHEMA
```
Building MIMIC may take a significant amount of time.

Now connect to the database
```bash
psql -U $DBUSER -d $DBNAME
```
To inspect the results:
```psql
set search_path to mimiciii, public;
\dt
```
shows you a list of tables in the database. If the list of empty (0 rows), something is wrong. Check the output of the previous step and ensure you are connected to the correct database and schema.

Finally, we have to derive data from the original MIMIC data. Derived tables are are referred to in MIMIC as 'concepts'. Exit psql, navigate to the `concepts_postgres` directory and connect to your database:
```bash
cd ../../concepts_postgres/
psql -U $DBUSER -d $DBNAME
```

Now build the concepts with the following commands:
```psql
set search_path to mimiciii, public;
\dt
\i postgres-functions.sql
\i postgres-make-concepts.sql
```
Deriving the MIMIC concepts may again take a significant amount of time, but not as long as building MIMIC.

Finally, create a Python3 virtual environment using [pyenv](https://virtualenv.pypa.io/en/latest/user_guide.html), [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) or your own favourite tool and install the dependencies in ``requirements.txt``.

## 2. Patient data selection
Once you have built MIMIC and derived the concepts you can run the scripts in the `sql` directory of the repository root to build a CSV of trajectory data, see `sql/README.md`.
By doing so, a single CSV with patient trajectories is created in the `/tmp/` directory. Change `to_csv.sql` to edit the target directory of this trajectory file.

Move the resulting trajectory CSV to the data directory:
```bash
mv /tmp/ventilatedpatients.csv $datadir
```

## 3. Clinically informed pre-processing
Run the notebook `preprocessing/Analysis and Preprocessing.ipynb` to create a train-val-test split for a single seed and perform clinically informed pre-processing. Run the notebook multiple times with varying seeds to implement random permutation cross-validation.

## 4. Data-driven pre-processing and policy training
To perform data-driven pre-processing for a single seed run the `scripts/Clustering and Training.ipynb` notebook. Run the notebook multiple times with varying seeds to implement random permutation cross-validation. Run all cells up to ``Training a policy``.

To train a policy, run ``scripts/policy_learning.py`` with the following parameters:

1. seed (int): the random seed
2. shaping (string): what type of shaping to use. Set to `avgpotential2` or `none`
3. compliance_scalar (float): a scalar to balance environment and shaping reward, referred to as 'c' in the paper
4. unsafety_prob (float): the probability of allowing a safety violation, currently only supports ``{0.0, 1.0}``. Set to `0.0` to enforce safety constraints during training, set to `1.0` to train an unconstrained policy.
5. softmax_temp (float, optional): temperature parameter for softmax, defaults to 1.0
6. plot (bool, optional): whether to generate plots, defaults to `False`


## 5. Off-policy evaluation
To perform off-policy evaluation for a single seed run `scripts/ope_script.py` for softmax policies and `scripts/ope_script_greedy.py` for greedy policies, both
with the following parameters:

1. seed (int): the random seed
2. shaping (string): what type of shaping to use. Set to `avgpotential2` or `none`
3. unsafety_prob (float): the probability of allowing a safety violation, currently only supports ``{0.0, 1.0}``. Set to `0.0` to enforce safety constraints during training, set to `1.0` to train an unconstrained policy.
4. shaping_scalar (float): a scalar to balance environment and shaping reward, referred to as 'c' in the paper
5. train_test (string): whether to perform OPE on the train or on the test set. Defaults to 'test'.


## 6. Analysis and evaluation
To aggregate all results and generate plots, use the notebook ``scripts/ope-visualisation-results.ipynb``.

To investigate the policies run ``scripts/Qualitative analysis.ipynb``.

# FAQ
**Q**: *Why is the `mimic-data` directory empty?*
**A**: This repository does not come with data, see [Getting started](#1.-Getting-started) to learn about access MIMIC.

**Q**: *Why is the `mimic-code` directory empty?*
**A**: You need to run `git submodule update` in the root directory of this repository to fetch the files in the `mimic-code` directory.
