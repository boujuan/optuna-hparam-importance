# Optuna Hyperparameter Importance Calculator

A utility for extracting and visualizing hyperparameter importance from Optuna studies using *fANOVA* analysis.

## Overview

This tool connects to an Optuna study stored in a PostgreSQL database, calculates the importance of each hyperparameter using fANOVA analysis, and generates both visualizations and CSV exports of the results.

![Hyperparameter Importance Example](example.png)

## Features

- Calculate hyperparameter importance using *fANOVA* [[CODE]](https://github.com/automl/fanova) [[PAPER]](https://proceedings.mlr.press/v32/hutter14.html)
- Generate bar plots for easy visualization of relative importance using *optuna-visualization* [[CODE]](https://github.com/optuna/optuna/blob/master/optuna/visualization/_param_importances.py) [[DOCS]](https://optuna.readthedocs.io/en/stable/reference/visualization/generated/optuna.visualization.plot_param_importances.html)
- Export raw importance values to CSV for further analysis
- Flexible database connection options

## Dependencies

Required Python packages:
- pandas
- matplotlib
- seaborn
- optuna
- psycopg2-binary
- sqlalchemy

## Installation

```bash
# Clone the repository
git clone https://github.com/boujuan/optuna-hparam-importance.git
cd optuna-hparam-importance

# Install dependencies
pip install pandas matplotlib seaborn optuna psycopg2-binary sqlalchemy -y

conda install pandas matplotlib seaborn optuna psycopg2-binary sqlalchemy -y

mamba install pandas matplotlib seaborn optuna psycopg2-binary sqlalchemy -y
```

## Usage

### Basic Usage

```bash
python calc_hyperparameter_importance.py -s "your_study_name" -p "your_password"
```

### Connect with a Database URL

```bash
python calc_hyperparameter_importance.py -s "your_study_name" --url "postgresql://username:password@hostname:port/database?sslmode=require"
```

### Command Line Arguments

| Argument | Short | Description |
|----------|-------|-------------|
| `--study-name` | `-s` | Name of the Optuna study |
| `--url` | | Complete database URL (overrides other DB options) |
| `--host` | `-a` | Database host |
| `--port` | | Database port |
| `--name` | `-db` | Database name |
| `--user` | `-u` | Database user |
| `--password` | `-p` | Database password |
| `--output-dir` | `-o` | Directory to save results |

## Output

The script creates two output files in the specified output directory:
- `{study_name}_param_importances.png`: Bar plot of parameter importance
- `{study_name}_param_importances.csv`: CSV file with importance values

## Post-Processing Results

For advanced analysis of the results, including combining importance data with Weights & Biases trials to identify efficient hyperparameter ranges, check out [this Claude artifact](https://claude.ai/public/artifacts/e9d4fd9b-aefa-48ef-b115-85bc2ae3d628?fullscreen=true).

