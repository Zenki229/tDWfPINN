# tDWfPINN: Transformed Diffusion-Waved Fractional PINNs

Refactored professional codebase for solving fractional PINNs using **Hydra** (Configuration), **Weights & Biases** (Experiment Tracking), and **Plotly** (Interactive Visualization).

## 🚀 Features
- **Hierarchical Configuration**: Manage experiments via `conf/` using Hydra.
- **Experiment Tracking**: Automatic logging of metrics, gradients, and artifacts to WandB.
- **Interactive Visualization**: Heatmaps and scatter plots with Plotly, saving raw data for reproducibility.
- **Modular Design**: Strict separation of Physics, Model, Data, and Training logic.
- **Reproducibility**: Global seeding and explicit version control logging.

## 📂 Directory Structure
```
tDWfPINN/
├── conf/              # Hydra Configuration
│   ├── config.yaml    # Main config
│   ├── model/         # Architecture settings
│   ├── pde/           # Physics parameters
│   └── experiment/    # Reproducible presets
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

### Basic Training
```bash
python src/train.py
```

### Debug Mode (Fast Run)
```bash
python src/train.py experiment=debug
```

### Overriding Parameters (Hydra Syntax)
```bash
# Change learning rate and max steps
python src/train.py optimizer.lr=0.005 training.max_steps=5000

# Change PDE alpha
python src/train.py pde.alpha=1.5
```

## 📊 Output
- **Logs**: WandB dashboard (online or offline).
- **Checkpoints**: Saved in `outputs/YYYY-MM-DD/HH-MM-SS/`.
- **Plots**: Interactive HTML and raw `.npz` data in `outputs/.../plots/` and `raw_data/`.

## 📐 Mathematical Notation
| Symbol | Meaning | Code Variable |
| :--- | :--- | :--- |
| $\alpha$ | Fractional Order | `pde.alpha` |
| $\lambda$ | Diffusion Coefficient | `pde.lambda_val` |
| $N$ | Batch Size (Domain) | `training.batch_size.domain` |

## 🧪 Testing
```bash
pytest tests/
```
