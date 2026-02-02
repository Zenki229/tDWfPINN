I will refactor the project into a professional, research-grade codebase using **Hydra**, **Weights & Biases (WandB)**, and **Plotly**, strictly adhering to the user's standards.

### **1. Directory Structure & Configuration Architecture**

* **`conf/`**: Hierarchical Hydra configs (`config.yaml`, `model/`, `pde/`, `experiment/`).

* **`src/`**: Modular source code.

  * `vis/`: **New** visualization module using Plotly.

  * `models/`, `physics/`, `data/`, `utils/`: Core logic modules.

* **`data/raw_plots/`**: Directory to store raw plot data for future restyling.

### **2. Technical Implementation Steps**

#### **Step 1: Infrastructure (Hydra + WandB)**

* Implement `src/utils/experiments.py` for global seeding and WandB setup.

* Create YAML configs to replace `ml_collections`.

#### **Step 2: Core Refactoring (OOP & Typing)**

* **Models**: Type-hinted `src/models/net.py` with Xavier init.

* **Data**: Seed-controlled `src/data/sampler.py`.

* **Physics**: `src/physics/dw_pde.py` implementing the fractional PINN logic with strict assertions.

#### **Step 3: Visualization (Plotly + Data Persistence)**

* **Replace Matplotlib**: Use `plotly.graph_objects` for interactive heatmaps and scatter plots.

* **Future-Proofing**: Implement a `save_plot_state(data_dict, filename)` function in `src/vis/plotter.py`.

  * Save the exact $(t, x, u, error)$ arrays as `.npz` or JSON.

  * This allows reloading the data later to adjust `font_size`, colormaps, etc., without retraining.

  * Log these artifacts to WandB.

#### **Step 4: Training & Verification**

* **Training Loop**: `src/train.py` with Hydra.

* **Validation**: Checkpoints, config validation, and convergence checks.

* **Tests**: Unit tests for math operators and config consistency.

### **3. Deliverables**

* Complete `conf/` and `src/` structure.

* `src/vis/` module for Plotly figures and raw data storage.

* Updated `README.md` with math symbols and reproduction steps.

