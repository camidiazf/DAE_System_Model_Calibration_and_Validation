# 🧪 DAE System Model Calibration and Validation

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![CasADi](https://img.shields.io/badge/CasADi-3.6-green.svg)](https://web.casadi.org/)
[![Jupyter](https://img.shields.io/badge/Notebook-Compatible-orange.svg)]()
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/camidiazf/DAE_System_Model_Calibration_and_Validation/blob/main/Main_Collab.ipynb)
[![License](https://img.shields.io/badge/license-MIT-lightgrey.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/platform-Linux%20%7C%20Windows%20%7C%20macOS-lightgrey.svg)]()


---

This repository implements a complete Python-based framework for the **calibration, analysis, and validation of models** expressed as **Differential–Algebraic Equations (DAEs)**.  
It was developed for modeling *Mycobacterium smegmatis* growth systems, and can be easily adapted to other restricted DAE systems.

---

## 🧩 Overview

The repository provides a reproducible and automated workflow to:

1. **Define and simulate** DAE-based bioprocess models using CasADi’s IDAS solver.  
2. **Calibrate parameters** against experimental data via **Particle Swarm Optimization (PSO)**.  
3. **Quantify uncertainty** using the **Fisher Information Matrix (FIM)**, **t-values**, and **correlation analysis**.  
4. **Perform sensitivity analysis** to assess parameter influence.  
5. **Generate publication-ready summaries and figures** for model validation.

The workflow is modular and can be executed from `Main.ipynb` or integrated into other computational pipelines.

---

## ⚙️ Pipeline Overview

The framework operates through the following stages:

```bash
DAE System Definition 
   ↓
PSO Parameter Calibration 
   ↓
Validation & Residual Analysis 
   ↓
Sensitivity & FIM Analysis 
   ↓
Summary Reports & Figures
```

### **1. Definition of the DAE System**

- Load DAE system's information and define different parameters for analysis in `System_data.py`.
- Load the experimental data as `Experimental_data_xlsx`.
- Defines the system’s differential and algebraic equations with **CasADi IDAS** in `DAE_Systems_Simulations.py`

### **2. Initial Validation  and Sensitivity Analysis**

- Through `RUN_INITIAL`.
- Computes statistical metrics such as **NRMSE**, **RMSE** and **MAPE**.  
- Derives **FIM**, **eigenvalues**,  and **correlation matrices** for identifiability.
- Perturbs parameters ±Δ to assess their local influence on system states.
- Produces bar charts and summary heatmaps.

### **3. Parameter Calibration**

- Uses **Particle Swarm Optimization (PSO)** to minimize the squared error between experimental and simulated data.
- Allows:
  - Multi-run iteration analysis (`RUN_PARAMETERS_ITERATIONS`)
  - Scenario-based parameter sweeps (`RUN_SCENARIO`)

### **4. Validation and Sentitivity Analysis of New Parameters**

- Re-do of step 2 but with calibrated parameters compared to the initial results.

### **5. Re-Calibration**

- Re-do of step 3 for as many times as user defines.

### **6. Summary & Visualization**

- Aggregates and visualizes results with mean ±σ across calibration runs.
- Generates structured folders containing Excel tables and figures.

---

## 📂 Repository Structure

```bash
.
├── DAE_analysis_package/               # MAIN SCRIPTS, DO NOT EDIT
│   ├── Aux_Functions.py                # Analytical utilities (FIM, residuals, sensitivities, correlations)
│   ├── Plotting_functions.py           # Visualization tools for summary figures and diagnostics
│   └── RUN_functions.py                # High-level orchestration (PSO, calibration loops, analysis aggregation)
│
├── M_smegmatis_O2/                     # CUSTOM CONTENT FOLDER FOR DAE SYSTEM
│   ├── DAE_Systems_Simulations.py      # DAE model definition and CasADi integration
│   ├── System_data.py                  # Experimental data import, parameters, bounds, solver config
│   ├── Experimental_data.xlsx          # Experimental dataset
│   ├── M_smegmatis_O2_Results/         # Auto-generated calibration & sensitivity outputs
│   │   ├── Calibration_1/              # Calibration run results
│   │   └── Calibration_2/              # Calibration run results
│
├── Main.ipynb                          # Main executable workflow (simulation → calibration → analysis)
├── Main_Colab.ipynb                   # Google Colab-compatible workflow version
├── Article.pdf                         # Associated research article draft
├── Supplementary_Material.pdf          # Extended data and results
├── requirements.txt                    # Python dependencies
└── README.md                           # Project documentation
```

---

## 🧠 Core Modules

These modules are global for all defined DAE systems, hence they are in a separate directory. They do not have to be modified for the package to function.

---

### `Aux_Functions.py`

Implements essential analytical routines for model calibration and validation:

- **`all_param_combos()`** — generates combinations of given parameters.
- **`validation_analysis()`** — performs validation analysis of the given model. Analysis includes residuals, RMSE, NRMSE, MAPE, AIC, BIC, and statistical tests on residuals.
- **`parameter_analysis()`** — performs parameter analysis of the given set.     Analysis includes sensitivity analysis, Fisher Information Matrix (FIM), parameter correlation matrix, and t-values.
- **`define_cost_function()`** — builds objective function for calibration using experimental data.
- **`sim_plus_minus()`** — simulates model with perturbed parameters by a defined percentage for sensitivity analysis or by a defined delta for FIM analysis.
- **`residuals_equations()`** — computes RMSE, MAPE, AIC, BIC for a simulation based on experimental validation values.  
- **`compute_FIM()`** — constructs Fisher Information Matrix using finite-difference Jacobians.  
- **`compute_correlation_matrix()`** — inverts FIM to estimate parameter correlations.  
- **`compute_t_values()`** — calculates t-values of calibrated parameters.
- **`compute_sensitivity()`** — performs local sensitivity analysis of each parameter.
- **`format_number()`** — formats number to a specific type.
- **`init_csv_with_header()`** — initializes a CSV file with a header.
- **`append_csv_row()`** —  adds a row to a csv file.

---

### `Plotting_functions.py`

Contains plotting and saving functions for analysis and visualization:

- **`_iter_index()`** — handles iteration indexing for plotting and saving figures.
- **`_ensure_dir()`** — ensures that the given directory exists.
- **`save_fig()`** — saves the figure to the given directory.
- **`figures_dir_from_csv_path()`** — generates the directory path to save figures.

- **Model analysis figures**: `plot_corr_matrix()`, `plot_sensitivity_analysis()`, `plot_residuals_analysis()` , `plotting_comparison()` — generate and save figures for parameter and model analysis.
- **Summary figures**: `plot_sensitivity_summary`,  `plot_residuals_summary`, `plot_corr_summary`— generate and save summary figures with mean ±σ values across calibration runs  .
- **Utility functions:** `_iter_index()`, `ensure_dir()`, `save_fig()`, `plot_corr_summary()` — handle iteration indexing, directory existence, figure saving and path creation.

---

### `RUN_functions.py`

Top-level orchestration of the calibration and analysis:

- **`suppress_all_output()`** — suppresses  all output from PSO Optimization to avoid cluttering the console during optimization.
- **`RUN_ALL_COMBOS()`** — begins calibration and analysis of all combinations given, returning a CSV file with results.
- **`RUN_PARAMETER_COMBO()`** — runs calibration and analysis of a given parameter combination or initial run, adding results to CSV file.
- **`RUN_INITIAL()`** — runs initial analysis with original parameter values. Point of comparison for all subsequent iterations.
- **`RUN_SCENARIO()`** — performs the parameter combination calibration for a given number of iterations.  
- **`RUN_PSO_CALIBRATION()`** — executes one PSO optimization and analysis.  
- **`RUN_ANALYSIS()`** — performs the validation and parameter analysis for the given parameters and values.  
- **`RUN_SUMMARY_ANALYSIS()`** — aggregates multi-run results and generates summary plots.  

---

## 🧠 Custom Modules

These modules are defined according to the user's DAE system. The names of the files must stay the same, only the name of this directory is custom (the model's name for user recognition such as `M_Smegmatis_O2`). Each file must be edited with the DAE system's information.

---

### `DAE_Systems_Simulations.py`

Defines and runs the dynamic DAE system:

- **`DAE_system()`** — defines system equations for state evolution and algebraic constraints.
- **`DAE_system_calibrating()`** — accepts parameter vectors for calibration mode of the system.

- **`simulate_model()`** — runs CasADi IDAS integration and outputs time-series DataFrames.  

---

### `System_data.py`

Contains static configurations:

- Experimental dataset import and preprocessing  
- Parameter bounds and units  
- Initial conditions and time grids  
- Solver tolerances and plotting parameters  

---

### `Experimental_data.xlsx`

Excel file that has at least two datasets: one for **calibration** or **parameter estimation** and one for **validation**. Instructions on its format are in `System_data.py`.

---

## 🚀 Getting Started

### **Installation**

Clone the repository and install dependencies:

```bash
git clone https://github.com/camidiazf/DAE_System_Model_Calibration_and_Validation.git
cd DAE_System_Model_Calibration_and_Validation
pip install -r requirements.txt
```

### **Run the Workflow**

Open the Jupyter notebook:

```bash
jupyter notebook Main.ipynb
```

Or run the Google Colab version:

```python
from google.colab import drive
drive.mount('/content/drive')
%run Main_Collab.ipynb
```

---

## 📊 Output Structure

Running the package for a set of parameters to calibrate, will create a result directory inside the folder of the defined DAE system. In this case `/M_smegmatis_O2/M_smegmatis_O2_Results/`.

Each time the package is run the user defines the variable `folder_calibration_results` which is the name of the current run. In this directory the results for the run will be saved. A CSV file with the results for the complete run, with all parameter combos is saved with the name of the DAE System name. In this case `Calibration_i` was used for different calibrations made.

For each calibration run, different combinations of parameters are used. Results for each combination results are saved in different directories. For example, `/[param1, param2]/`, will store the results and figures for the calibration of `param1` and `param2`. The `/Original/` directory stores results for the original model with its original values.

Inside each of the parameter results directory there will be a correlation matrix, a comparison plot, a residual analysis and sensitivity analysis figure for all iterations, including the summary figures of all iterations.

```bash
├── M_smegmatis_O2/                     # DAE system folder
│   ├── M_smegmatis_O2_Results/         # Auto-generated results for DAE system
│   │   ├── Calibration_1/              # Calibration 1 run results
│   │   │   ├── ['Xmax']/               # Parameter combination results
│   │   │   ├── ['YX_N','Xmax']/        
│   │   │   ├── ['YX_C', 'YX_N', 'mu_max', 'pH_UL']/ 
│   │   │   └── ['pH_UL']/ 
│   │   │   │   ├── Corr_Matrix_1.png/
│   │   │   │   ├── Corr_Matrix_i.png/ 
│   │   │   │   ├── Residuals_Analysis_1.png/ 
│   │   │   │   ├── Residuals_Analysis_i.png/ 
│   │   │   │   ├── Sensitivity_1.png/ 
│   │   │   │   ├── Sensitivity_i.png/ 
│   │   │   │   ├── Plotting_Comparison_1.png/ 
│   │   │   │   ├── Plotting_Comparison_i.png/ 
│   │   │   │   ├── Corr_Matrix_summary.png/ 
│   │   │   │   ├── Residuals_Analysis_summary.png/ 
│   │   │   │   └── Sensitivity_summary.png/ 
│   │   │   └── Original/ 
│   │   │   │   ├── Corr_Matrix_original.png/ 
│   │   │   │   ├── Residuals_Analysis_original.png/ 
│   │   │   │   ├── Sensitivity_original.png/ 
│   │   │   │   └── Plotting_Comparison_original.png/ 
│   │   ├── Calibration_2/
│   │   │   ├── ...  

```

---

## 📄 Associated Article and Supplementary Material

This repository includes the complete **research article** *(unpublished)* and **supplementary material** describing the modeling, calibration, and analysis of *Mycobacterium smegmatis* growth.

- **Article:** [`Sensitivity-Driven Optimization of a Batch Cultivation DAE Model of Mycobacterium smegmatis.pdf`](./Article.pdf)  
- **Supplementary Material:** [`Supplementary_Material.pdf`](./Supplementary_Material.pdf)

These documents provide:

- The biological and mathematical background of the DAE model.  
- Details of the calibration experiments, datasets, and validation results.  
- Discussion of parameter identifiability and sensitivity outcomes obtained using this framework.

---

## 🧾 Citation

If you use this framework, please cite:

> **Díaz-Figueroa, C.** (2025). *DAE System Model Calibration and Validation: A Python-based Framework for Bioprocess Analysis.*  
> GitHub Repository: [https://github.com/camidiazf/DAE_System_Model_Calibration_and_Validation](https://github.com/camidiazf/DAE_System_Model_Calibration_and_Validation)

---

© 2025 Camila Díaz Figueroa — MIT License.
