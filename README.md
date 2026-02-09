# CounterFactualDPG

A counterfactual explanation framework built on top of DPG. This project implements counterfactual generation methods for machine learning models, providing tools for explanation generation, evaluation, and visualization across multiple datasets.

## Project Relations

- **Parent Project**: [DPG](https://github.com/Meta-Group/DPG) - Direct Policy Gradient optimization framework
- **Sibling Project**: [DPG-augmentation](https://github.com/Meta-Group/DPG-augmentation) - Data augmentation techniques for DPG

## Overview

This repository provides a framework for generating and analyzing counterfactual explanations for machine learning models. The implementation focuses on practical counterfactual generation with proper handling of feature constraints, validation mechanisms, and evaluation metrics across multiple benchmark datasets.

### Core Components

- **CounterfactualModel**: Main implementation of counterfactual generation algorithms
- **ConstraintParser**: Framework for defining and handling feature constraints
- **ConstraintScorer** & **ConstraintValidator**: Tools for enforcing validity of generated counterfactuals
- **CounterFactualMetrics**: Evaluation metrics for assessing counterfactual quality
- **CounterFactualVisualizer**: Visualization tools for analyzing counterfactual explanations
- **HeuristicRunner**: Execution framework for running counterfactual experiments


## Installation

```bash
# Create a virtual environment
python -m venv .venv

# Activate the virtual environment
source .venv/bin/activate

# Install dependencies
pip install -r ./requirements.txt
```

## Optional: Clean Notebook Outputs

nbstripout is recommended to keep notebooks clean of output cells. Install globally and enable as a git hook:

```bash
pip install nbstripout
nbstripout --install
```