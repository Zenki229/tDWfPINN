#!/bin/bash 

# the script for running the debug example 1, for example 1, some configs are configured at the __init__ method.
ALPHA=1.25 
method='MC-I'
plot_backend='matplotlib'



export PYTHONPATH=$PYTHONPATH:.
python src/train.py \
    +experiment=debug \
    pde=dw_eg1 \
    pde.alpha=${ALPHA} \
    pde.method=${method} \
    wandb.project="debug-eg1" \
    plot=${plot_backend} 
