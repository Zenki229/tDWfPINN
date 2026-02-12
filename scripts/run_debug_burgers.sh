#!/bin/bash 

# the script for running the debug example 1, for example 1, some configs are configured at the __init__ method.
ALPHA=1.25 
method='GJ-I'
plot_backend='matplotlib'

# Set PYTHONPATH
export PYTHONPATH=$PYTHONPATH:.

# Run training in debug mode
# Using small epochs and steps for quick verification
python src/train.py \
    +experiment=debug \
    pde=burgers \
    pde.alpha=1.1 \
    pde.datafile='data/burgers_110.npz' \
    pde.method=${method} \
    wandb.project="debug-burgers" \
    wandb.name=${method} \
    plot=${plot_backend} 
