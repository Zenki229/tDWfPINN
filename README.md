# tDWfPINN: Transformed Diffusion-Waved Fractional PINNs

Refactored professional codebase for solving fractional PINNs using **Hydra** (Configuration), **Weights & Biases** (Experiment Tracking), and **Plotly** (Interactive Visualization). See Paper here. 📄 <a href="https://arxiv.org/abs/2506.11518">Paper</a> 

## TODO 
1. **[FINISH]** add unit-test for computing **MC-I, MC-II, GJ-I, GJ-II** for PDEs.
2. **[FINISH]** Improving plotting
3. In the future, we will update the code to JAX.
## 🚀 Features
- **Hierarchical Configuration**: Manage experiments via `conf/` using Hydra.
- **Experiment Tracking**: Automatic logging of metrics, gradients, and artifacts to WandB.
- **Interactive Visualization**: Heatmaps and scatter plots with Plotly OR MatPlotLib, saving raw data for reproducibility.
- **Modular Design**: Strict separation of Physics, Model, Data, and Training logic.
- **Reproducibility**: Global seeding and explicit version control logging.
  

## 📂 Directory Structure
```
tDWfPINN/
├── conf/              # Hydra Configuration
│   ├── config.yaml    # Main config
│   ├── model/         # Architecture settings
│   ├── pde/           # Physics parameters
│   ├── optimizer/     # Optimizer settings
│   ├── plot/          # Plotting settings
│   └── experiment/    # Debug presets
├── src/               # Source Code
│   ├── data/          # Samplers (TimeSpaceSampler)
│   ├── models/        # Neural Networks (MLP)
│   ├── physics/       # PDE definitions & Fractional Ops
│   ├── vis/           # Plotly Visualization
│   └── train.py       # Main Entry Point
└── tests/             # Unit Tests
```

## 🛠 Installation
```bash
pip install -r requirements.txt
# Ensure pymittagleffler is installed
```

## 🏃 Usage
All running scripts are in `scripts/`. You must preset the `wandb` setting before training.
### Debug Mode (Fast Run)
```bash
bash scripts/run_debug_eg1.sh
```

### EG1
```bash
bash scripts/run_eg1.sh
```

## 📊 Output
- **Logs**: WandB dashboard (online or offline).
- **Checkpoints**: Saved in `outputs/${wandb.project}-${wandb.name}/${now:%Y-%m-%d_%H-%M-%S}/`.
- **Plots**: Interactive HTML(plotly) or JPG and raw `.npz` data in the same output directory under `plots/` and `raw_data/`.

## 🧪 Testing
```bash
pytest tests/
```
