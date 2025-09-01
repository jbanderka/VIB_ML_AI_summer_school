# Lightning quickstart

## Overview

This is a quickstart repo for PyTorch Lightning.
See [the Lightning docs](https://lightning.ai/docs/pytorch/stable/) for more details.

## Project structure

```sh
lightning_quickstart/
├── config.yaml
├── project
│   ├── data.py
│   └── model.py
├── README.md
├── setup.py
```

## Installation

```sh
mamba create -n project python=3.11
mamba activate project
pip install lightning torch tqdm 'jsonargparse[signatures]' torchmetrics transformers
pip install -e .
```

## Training

```sh
python project/model.py fit -c config.yaml --trainer.logger.name=my_run_name
```
