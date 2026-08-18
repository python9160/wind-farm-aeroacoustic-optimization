# Wind Farm Layout Optimization with Aeroacoustic Validation

This project develops a workflow for optimizing wind farm layouts while accounting for both power production and aeroacoustic constraints.

The core optimization is performed with a **Genetic Algorithm (GA)**. To make repeated fitness evaluation practical, the GA uses:

- **FLORIS** for wake modeling and turbine power estimation.
- A **machine-learning aeroacoustic surrogate** for fast receiver-level noise prediction.
- **OpenFAST / FAST.Farm** as a higher-fidelity validation step for the best layout produced by the GA.

The overall goal is to avoid running OpenFAST during every GA fitness evaluation while still validating the final optimized solution against OpenFAST aeroacoustic output.

---

## Table of Contents

- [Installation](#installation)
  - [Step 1: Install OpenFAST Executables](#step-1-install-openfast-executables)
  - [Step 2: Install `openfast_toolbox`](#step-2-install-openfast_toolbox)
  - [Step 3: Configure Jupyter](#step-3-configure-jupyter)
  - [Step 4: Install Python Dependencies](#step-4-install-python-dependencies)
  - [Step 5: Check CPU and Parallelization Settings](#step-5-check-cpu-and-parallelization-settings)
  - [Step 6: Configure the Optimization](#step-6-configure-the-optimization)
  - [Step 7: Verify the Installation](#step-7-verify-the-installation)
  - [Step 8: Run the Genetic Algorithm](#step-8-run-the-genetic-algorithm)
  - [Step 9: Run OpenFAST / FAST.Farm Validation](#step-9-run-openfast--fastfarm-validation)
  - [Step 10: Review the Results](#step-10-review-the-results)
  - [Optional: Export the Notebook to Python](#optional-export-the-notebook-to-python)
- [Project Workflow](#project-workflow)
  - [1. Aeroacoustic surrogate development](#1-aeroacoustic-surrogate-development)
  - [2. Genetic Algorithm layout optimization](#2-genetic-algorithm-layout-optimization)
  - [3. OpenFAST / FAST.Farm validation](#3-openfast--fastfarm-validation)
- [Repository Structure](#repository-structure)
- [Wrapper Package](#wrapper-package)
  - [Utility Groups](#utility-groups)
- [Important Data Files](#important-data-files)
  - [`training_data.csv`](#training_datacsv)
  - [`noise_lookup_table.json`](#noise_lookup_tablejson)
- [Main Notebooks](#main-notebooks)
  - [`final_train.ipynb`](#final_trainipynb)
  - [`GA_with_OpenFAST_validation.ipynb`](#ga_with_openfast_validationipynb)
- [Fitness Function](#fitness-function)
  - [Aeroacoustic Aggregation](#aeroacoustic-aggregation)
- [Aeroacoustic Model Configuration](#aeroacoustic-model-configuration)
- [OpenFAST Computational Domain](#openfast-computational-domain)
- [Parallelization](#parallelization)
- [Requirements](#requirements)
- [Notes](#notes)
- [Possible Future Work](#possible-future-work)
- [Status](#status)

---

## Installation

### Step 1: Install OpenFAST Executables

Download the required executables from the [OpenFAST Official Repository](https://github.com/OpenFAST/openfast/releases).

The primary executables required by this project are:

```text
FAST.Farm.exe
OpenFAST.exe
```

Place both executables in the root directory of this project (`./`). `wrapper.main` will look for them at this location using these filenames.

If you would also like to generate turbulent wind fields, download `TurbSim.exe` and place it in the project root as well. The current workflow uses steady-state wind for OpenFAST validation, so TurbSim is not required. However, the `run_turbsim` utility is available in `wrapper.main` for future use.

---

### Step 2: Install `openfast_toolbox`

Download the official [`openfast_toolbox`](https://github.com/OpenFAST/openfast_toolbox).

The package is not currently distributed through PyPI, so follow the **Installation and testing** instructions provided in the `openfast_toolbox` README.

Make sure that `openfast_toolbox` is installed into the same Python environment that will be used to run this project.

---

### Step 3: Configure Jupyter

Ensure that Jupyter Notebook support is installed. This requires both a notebook interface and a Python kernel.

For Visual Studio Code, install the official [Jupyter extension](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter) and install the required Python package:

```bash
pip install notebook
```

or, if only a Jupyter kernel is required:

```bash
pip install ipykernel
```

When opening the project notebooks, make sure the selected Jupyter kernel corresponds to the Python environment in which the project dependencies are installed.

---

### Step 4: Install Python Dependencies

Ensure that the following Python packages are installed:

```text
numpy
pandas
matplotlib
scipy
scikit-learn
tqdm
joblib
floris
```

They can be installed or updated together with:

```bash
pip install -U numpy pandas matplotlib scipy scikit-learn tqdm joblib floris
```

Make sure these packages are installed in the same Python environment used by the Jupyter kernel and `openfast_toolbox`.

---

### Step 5: Check CPU and Parallelization Settings

FLORIS evaluations are parallelized across candidate farm layouts during the Genetic Algorithm. The appropriate number of worker processes depends on the CPU running the optimization.

Run the supplied [`core_checker.py`](core_checker.py) from the project root:

```bash
python core_checker.py
```

The utility will inspect the available CPU resources and provide recommended starting values for:

```python
MAX_WORKERS
FLORIS_CHUNKSIZE
```

Copy the recommended values into the parallelization settings in [`GA_with_OpenFAST_validation.ipynb`](GA_with_OpenFAST_validation.ipynb).

For example:

```python
MAX_WORKERS = 6
FLORIS_CHUNKSIZE = 4
```

`MAX_WORKERS` controls the maximum number of FLORIS worker processes that can execute concurrently.

`FLORIS_CHUNKSIZE` controls how population evaluations are grouped when work is distributed to the FLORIS worker processes.

The values reported by `core_checker.py` should be treated as starting recommendations rather than strict requirements. The optimal configuration depends on the processor architecture, available memory, and other workloads running on the system.

In particular, CPUs with hybrid architectures containing separate performance and efficiency cores may require additional manual tuning. If the system becomes unresponsive or memory usage becomes excessive during optimization, reduce `MAX_WORKERS`.

---

### Step 6: Configure the Optimization

Open [`GA_with_OpenFAST_validation.ipynb`](GA_with_OpenFAST_validation.ipynb) and review the configuration cells near the beginning of the notebook.

Configure the required settings before starting the optimization, including:

- farm dimensions and boundary constraints;
- number of turbines;
- minimum turbine spacing;
- receiver locations;
- wind speeds and their associated weights;
- wind directions and their associated weights;
- FLORIS turbulence intensity;
- GA population size and number of generations;
- crossover and mutation settings;
- immigration settings;
- acoustic noise limit;
- fitness penalty coefficients;
- `MAX_WORKERS`;
- `FLORIS_CHUNKSIZE`; and
- OpenFAST / FAST.Farm validation settings.

Wind-speed and wind-direction distributions are combined as a Cartesian product to create the complete set of operating scenarios. Each resulting scenario is assigned a joint probability based on its wind-speed and wind-direction weights.

The scenario weights should collectively represent the desired wind distribution and sum to `1.0`.

---

### Step 7: Verify the Installation

Before starting a potentially long optimization run, execute the supplied [`smoke_test.py`](smoke_test.py) from the project root:

```bash
python smoke_test.py
```

The smoke test verifies that:

- the required Python dependencies can be imported;
- `openfast_toolbox` is available;
- `OpenFAST.exe` and `FAST.Farm.exe` are present;
- the `5MW`, `5MW_farm`, `wrapper`, and `dba_model_outputs` directories are available;
- the main `wrapper` modules can be imported;
- the FLORIS `gch.yaml` configuration is present;
- a trained `.joblib` aeroacoustic surrogate model can be located;
- `training_data.csv` can be read; and
- the expected ML training variables are present.

`TurbSim.exe` and other optional development files will produce warnings rather than cause the smoke test to fail.

A successful installation should end with:

```text
Smoke test PASSED.

The required project files, Python dependencies, OpenFAST executables,
and wrapper modules were found.

You should be ready to configure and run
GA_with_OpenFAST_validation.ipynb.
```

If one or more required components cannot be located or imported, the test will instead report:

```text
Smoke test FAILED.
```

Review the `[FAIL]` entries reported by the utility before proceeding.

The smoke test does **not** execute OpenFAST, FAST.Farm, FLORIS simulations, or the Genetic Algorithm. It is intended only to verify the project environment before starting a computationally expensive run.

---

### Step 8: Run the Genetic Algorithm

Once the smoke test passes, open [`GA_with_OpenFAST_validation.ipynb`](GA_with_OpenFAST_validation.ipynb) using the configured Jupyter environment.

Run the notebook from beginning to end.

The first portion of the notebook performs the Genetic Algorithm optimization. Candidate layouts are evaluated using:

```text
Candidate Layout
       │
       ├──> FLORIS
       │      └──> Farm Power + Waked Turbine Velocities
       │
       └──> ML Aeroacoustic Surrogate
              └──> Receiver dBA
                       │
                       ▼
                  Fitness Function
```

FLORIS evaluates wake interactions and turbine power, while the vectorized ML AeroAcoustic surrogate predicts the acoustic contribution from each turbine to each receiver.

These evaluations are performed across the configured wind scenarios and combined using their associated scenario weights.

The GA then applies the configured spacing, boundary, and acoustic penalties to determine the fitness of each chromosome.

The optimization continues for the configured number of generations while retaining the best-performing layout.

---

### Step 9: Run OpenFAST / FAST.Farm Validation

After the GA completes, continue running the remaining cells in [`GA_with_OpenFAST_validation.ipynb`](GA_with_OpenFAST_validation.ipynb).

The best chromosome found by the GA is evaluated using the higher-fidelity OpenFAST / FAST.Farm workflow.

For each configured wind scenario, the validation workflow:

1. rotates the optimized layout into the OpenFAST coordinate frame;
2. calculates the required FAST.Farm low-resolution computational domain;
3. configures the corresponding OpenFAST / FAST.Farm input files;
4. runs the farm simulation;
5. collects the AeroAcoustics output from each turbine;
6. removes the initial transient portion of the simulation;
7. calculates the steady-state energetic mean (`L_eq`) for each turbine-observer pair;
8. energetically combines the individual turbine contributions into total farm-level observer dBA;
9. compares the OpenFAST results against the ML surrogate predictions; and
10. recalculates the acoustic fitness using the OpenFAST results.

The validation portion also generates comparison tables and visualizations for evaluating agreement between the surrogate and OpenFAST results.

> **Note:** The current OpenFAST validation workflow uses deterministic steady-state wind and assumes 0% turbulence intensity. Turbulent OpenFAST inflow would require generating appropriate wind fields using TurbSim and is not part of the default validation workflow.

---

### Step 10: Review the Results

After the notebook has completed, review both the GA optimization results and the subsequent OpenFAST validation.

The final outputs include information such as:

- optimized turbine coordinates;
- expected farm power;
- maximum boundary receiver dBA;
- scenario-specific power and acoustic results;
- weighted wind-scenario performance;
- FLORIS/AASurrogate fitness;
- OpenFAST-validated acoustic performance;
- surrogate vs. OpenFAST receiver-level comparisons;
- spatial acoustic error distributions; and
- comparison visualizations.

The OpenFAST validation results should be used to determine whether the optimized layout identified using the surrogate maintains acceptable agreement when evaluated using the higher-fidelity AeroAcoustics model.

---

### Optional: Export the Notebook to Python

The complete optimization and validation driver is currently implemented as a Jupyter notebook.

If a standalone `.py` driver is preferred, [Visual Studio Code provides tools for exporting a Jupyter notebook to a Python script](https://code.visualstudio.com/docs/datascience/jupyter-notebooks#_export-your-jupyter-notebook).

The resulting Python script can then be modified as needed for batch execution or integration into another workflow.

---

## Project Workflow

### 1. Aeroacoustic surrogate development

OpenFAST aeroacoustic results are used to build a training dataset containing quantities such as:

- turbine effective wind speed
- receiver distance
- receiver angle relative to wind direction
- A-weighted sound pressure level

The current ML surrogate uses:

```text
v
log10(r)
cos(theta)
sin(theta)
```

to predict receiver-level `dBA`.

Training, model comparison, diagnostics, and visualizations are contained in:

```text
final_train.ipynb
```

Generated model artifacts and model-comparison visualizations are stored in:

```text
dba_model_outputs/
```

### 2. Genetic Algorithm layout optimization

The main optimization notebook is:

```text
GA_with_OpenFAST_validation.ipynb
```

This notebook contains two major parts:

1. The GA optimization itself.
2. OpenFAST / FAST.Farm validation of the best chromosome found by the GA.

The optimizer uses:

- the vectorized ML aeroacoustic surrogate for receiver noise prediction
- FLORIS for wake interactions, effective turbine wind speeds, and farm power
- weighted wind-speed and wind-direction scenarios
- turbine-spacing constraints
- farm-boundary setback constraints
- acoustic noise constraints

The GA evaluates candidate layouts under a Cartesian product of wind-speed and wind-direction scenarios. Scenario probabilities are used to calculate weighted performance.

### 3. OpenFAST / FAST.Farm validation

After the GA finishes, the best layout is validated using OpenFAST / FAST.Farm.

For each wind scenario:

1. The optimized layout is rotated into the OpenFAST coordinate frame.
2. A FAST.Farm low-resolution computational domain is generated automatically from the rotated turbine layout.
3. A deterministic uniform inflow is used.
4. OpenFAST aeroacoustic observer outputs are collected.
5. The second half of the simulation is used as the steady-state region.
6. Receiver sound levels are energetically combined across turbines.
7. OpenFAST receiver dBA values are compared against the ML surrogate predictions.
8. The GA fitness is recalculated using OpenFAST acoustics in place of surrogate acoustics.

OpenFAST validation currently assumes **0% turbulence intensity** because turbulent inflow would require running TurbSim for each case, which is significantly more expensive. The GA itself may still use a non-zero turbulence intensity in FLORIS.

---

# Repository Structure

```text
.
├── 5MW/
│   └── OpenFAST turbine model files
│
├── 5MW_farm/
│   └── FAST.Farm model files and supporting configuration
│
├── dba_model_outputs/
│   ├── trained ML model artifacts
│   ├── model comparison results
│   └── validation / diagnostic visualizations
│
├── wrapper/
│   ├── __init__.py
│   ├── AASurrogate.py
│   ├── AASurrogateCorr.py
│   ├── aa_surrogate_ml.py
│   ├── aa_surrogate_ml_vectorized.py
│   ├── examples.py
│   ├── farm.py
│   ├── floris.py
│   ├── gch.yaml
│   ├── main.py
│   ├── openfast_base.py
│   ├── openfast_value.py
│   ├── parallel_floris.py
│   ├── parallel_floris_class.py
│   └── utils.py
│
├── core_checker.py
├── smoke_test.py
├── final_train.ipynb
├── GA_with_OpenFAST_validation.ipynb
├── noise_lookup_table.json
├── training_data.csv
└── README.md
```

---

# Wrapper Package

`wrapper` is a collection of utilities, I/O managers, surrogate models, and driver code used throughout the project.

## Utility Groups


| Module                                                                | Purpose                                                                           |
| --------------------------------------------------------------------- | --------------------------------------------------------------------------------- |
| [`openfast_base`](wrapper/openfast_base.py)                           | Provides`OpenFASTFile` and related OpenFAST file-management functionality.        |
| [`utils`](wrapper/utils.py)                                           | Miscellaneous reusable utilities used throughout the project.                     |
| [`main`](wrapper/main.py)                                             | Driver functions for OpenFAST, FAST.Farm, and TurbSim execution.                  |
| [`farm`](wrapper/farm.py)                                             | `.fstf` I/O and FAST.Farm management built on NREL's official `openfast_wrapper`. |
| [`AASurrogate`](wrapper/AASurrogate.py)                               | Original/base aeroacoustic surrogate implementation.                              |
| [`AASurrogateCorr`](wrapper/AASurrogateCorr.py)                       | Corrected/extended version of the original aeroacoustic surrogate.                |
| [`aa_surrogate_ml_vectorized`](wrapper/aa_surrogate_ml_vectorized.py) | Vectorized machine-learning aeroacoustic surrogate used by the GA.                |
| [`parallel_floris`](wrapper/parallel_floris.py)                       | FLORIS parallel-evaluation utilities.                                             |
| [`parallel_floris_class`](wrapper/parallel_floris_class.py)           | Persistent FLORIS evaluation class with context-manager support.                  |

The persistent FLORIS evaluator is intended to avoid repeatedly creating worker processes and reinitializing FLORIS models during every GA generation.

---

# Important Data Files

## [`training_data.csv`](training_data.csv)

Primary training dataset used by the ML aeroacoustic surrogate.

The current model uses:

```text
v
r
cos_theta
sin_theta
dBA
```

with `log10(r)` derived during preprocessing.

## [`noise_lookup_table.json`](noise_lookup_table.json)

Legacy aeroacoustic lookup-table data used by earlier surrogate implementations.

This remains in the repository for comparison and compatibility with older workflows.

---

# Main Notebooks

## [`final_train.ipynb`](final_train.ipynb)

Contains the ML surrogate development workflow, including:

- loading and cleaning training data
- train/test splitting
- feature engineering
- comparison of multiple regression approaches
- evaluation using MAE, RMSE, and R²
- residual analysis
- prediction-vs-actual plots
- model serialization
- saved comparison visualizations

## [`GA_with_OpenFAST_validation.ipynb`](GA_with_OpenFAST_validation.ipynb)

Main optimization and validation notebook.

The notebook involves two main parts: the GA optimization itself, followed by OpenFAST validation of the best chromosome.

The optimizer uses the vectorized ML aeroacoustic surrogate together with FLORIS to replace repeated OpenFAST execution during the GA.

All GA, environmental, penalty, wind-scenario, FLORIS, and OpenFAST settings are defined in the configuration cells near the beginning of the notebook.

---

# Fitness Function

The GA maximizes a fitness function based on expected farm power minus penalties.

```text
fitness =
    expected power
    - turbine spacing penalty
    - acoustic noise penalty
    - boundary/setback penalty
```

Expected power is calculated as the weighted average of FLORIS power across all wind scenarios.

Acoustic penalties may be calculated using either:

- weighted scenario-specific violations, or
- the worst-case scenario

depending on the selected noise-penalty mode.

Because dBA is logarithmic, receiver levels across probabilistic scenarios are combined energetically rather than by directly averaging dBA values.

---

### Aeroacoustic Aggregation

FAST.Farm does not provide a farm-wide implementation of the OpenFAST AeroAcoustics (AA) module. Therefore, the acoustic contribution from each turbine is evaluated independently, and the resulting sound levels are combined in the linear acoustic-energy domain.

For \(N\) independent turbine contributions at a given observer, the total sound level is calculated as:

```text
L_total = 10 * log10(sum(10^(L_i / 10)))
```

For time-varying OpenFAST AA output, the steady-state sound level for each turbine-observer pair is calculated using the energetic mean:

```text
L_eq = 10 * log10(mean(10^(L_i / 10)))
```

In the current validation workflow, \(L_{eq}\) is calculated from the OpenFAST AA results after the initial transient period (`time > 15 s`).

The same farm-level aggregation approach is used for both the **FLORIS/AASurrogate optimization workflow** and the **FAST.Farm/OpenFAST validation workflow**. For each turbine, the AA observer coordinates are transformed into that turbine's relative coordinate frame, allowing its individual contribution to each observer to be evaluated.

With the AASurrogate, each turbine-observer pair produces a direct steady-state dBA prediction. With OpenFAST AA, the output is time-varying, so \(L_{eq}\) is first calculated over the selected steady-state period. In either case, the result can be represented as an \((N, M)\) matrix, where \(N\) is the number of turbines and \(M\) is the number of observers:

```text
                         Observer
                   1       2       ...      M
Turbine 1        L_11    L_12      ...    L_1M
Turbine 2        L_21    L_22      ...    L_2M
   ...            ...     ...      ...     ...
Turbine N        L_N1    L_N2      ...    L_NM
```

The individual turbine contributions are then energetically summed along the turbine dimension using \(L_{total}\), reducing the \((N, M)\) matrix to an \((M,)\) vector containing the total farm-level dBA at each observer.

This makes the two evaluation paths directly comparable:

```text
AASurrogate:  (N, M) steady-state predictions ──┐
                                                ├── L_total ──> (M,) farm dBA
OpenFAST AA:  time series ──> L_eq ──> (N, M) ──┘
```

Here, \(L_{eq}\) performs the **time-domain reduction** for each individual turbine-observer pair, while \(L_{total}\) performs the **farm-level aggregation across turbines** for each observer.

---

# Aeroacoustic Model Configuration

For the OpenFAST AeroAcoustics calculations, the turbulent inflow noise model is configured with:

```text
TIMod = 1
```

This selects the **Amiet model**. `TIMod = 2`, which combines the **Amiet model with the Simplified Guidati model**, was also tested during development. However, when used with the NREL 5 MW turbine configuration in this project, the predicted sound levels increased to clearly nonphysical magnitudes. The highest finite value observed was approximately **385.3 dBA**, after which OpenFAST began reporting the resulting sound levels as `Infinity`.

This behavior suggests a numerical instability or overflow occurring somewhere in the Simplified Guidati contribution. A division by zero or a near-zero quantity is one possible cause, but the exact source has not been identified. The behavior may instead be related to the blade configuration, operating conditions, model inputs, or another numerical condition within the implementation and could be investigated further in future work.

Because these nonphysical and eventually non-finite values prevented reliable acoustic aggregation and comparison with the surrogate model, `TIMod = 1` was used for the OpenFAST validation workflow.

This should be considered an implementation/modeling choice for the current study rather than an indication that `TIMod = 2` is generally invalid. Further investigation would be required to determine the exact input, environmental condition, blade configuration, or numerical behavior responsible for the observed results.

The complete set of aeroacoustic conditions can be viewed here: [Single-Turbine Configuration](5MW/AA.dat) / [FAST.Farm Configuration](5MW_farm/base_aa.dat)

The full description of the supplied aeroacoustic settings can be read [on the OpenFAST docs](https://openfast.readthedocs.io/en/dev/source/user/aerodyn-aeroacoustics/index.html). [Section 4.5.4](https://openfast.readthedocs.io/en/dev/source/user/aerodyn-aeroacoustics/index.html) goes into detail on the specifics of `TIMod`.

# OpenFAST Computational Domain

The OpenFAST / FAST.Farm validation code dynamically calculates the low-resolution FAST.Farm computational domain for each rotated layout.

The domain includes:

- upstream clearance
- larger downstream wake clearance
- crosswind clearance
- vertical rotor clearance

Grid dimensions are generated automatically from the turbine bounding box and configured grid spacing.

This avoids relying on a single hard-coded low-resolution domain for layouts and wind directions that may vary significantly.

---

# Parallelization

FLORIS is the primary computational bottleneck during optimization.

The project parallelizes FLORIS evaluation across candidate layouts.

A persistent worker-pool implementation is provided so that FLORIS worker processes and model instances can remain alive across GA generations.

The aeroacoustic ML surrogate is separately vectorized across:

- layouts
- wind scenarios
- turbines
- receivers

---

# Requirements

Major dependencies include:

```text
numpy
pandas
scipy
matplotlib
tqdm
scikit-learn
joblib
floris
openfast / FAST.Farm (executable)
openfast_wrapper
```

The exact versions should ideally be recorded in a project environment file once the workflow is finalized.

---

# Notes

- OpenFAST validation currently uses deterministic uniform inflow and therefore does not reproduce the FLORIS turbulence-intensity setting.
- TurbSim can be used for turbulent OpenFAST inflow, but this is intentionally excluded from routine validation because of its computational cost.
- The ML surrogate is intended to accelerate optimization, not replace OpenFAST as the final higher-fidelity validation model.
- `noise_lookup_table.json` represents an earlier surrogate approach and is retained as historical/reference data.
- The GA and OpenFAST validation code assumes consistent wind-direction conventions between FLORIS, the surrogate, and the layout-rotation utilities.

---

# Possible Future Work

Potential extensions include:

- validating additional turbulence conditions using pre-generated TurbSim fields
- adding uncertainty bounds to the ML acoustic surrogate
- multi-objective optimization of power and noise rather than a single penalized fitness value
- expanding the wind rose to measured site-specific distributions
- incorporating additional atmospheric absorption or terrain effects
- comparing multiple turbine models
- automated OpenFAST validation of several top GA chromosomes rather than only the best layout
- adding formal experiment/configuration tracking
- saving and restoring GA populations between runs

---

# Status

This project is currently under active development.

The current workflow is designed primarily as a research and optimization framework for comparing fast surrogate-based wind-farm aeroacoustic optimization against higher-fidelity OpenFAST / FAST.Farm validation.
