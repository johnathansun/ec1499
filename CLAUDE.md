# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This repository contains both a replication of Bernanke and Blanchard (2023) "What Caused U.S. Pandemic-Era Inflation?" and an **extended "New Model"** that modifies the original SVAR framework. The package recreates empirical results, figures, and tables using a multi-language econometric analysis workflow.

## New Model vs. Original Bernanke-Blanchard

The repository includes an alternative model specification (see `new_model.tex`) with the following key changes:

### 1. Wage Equation: Added Capacity Utilization
**Original BB:** Wages depend on expected prices, aspirational real wage, and labor market tightness (V/U).

**New Model:** Adds **capacity utilization** as a driver of wages. Rationale: when manufacturing operates at higher capacity, it puts upward pressure on wages beyond labor market tightness alone.

### 2. Price Equation: Endogenized Shortages
**Original BB:** Shortages are treated as **exogenous** (measured via Google Trends).

**New Model:** Shortages are now **endogenous** and decomposed:
```
log(s_t) = φ·log(s_{t-1}) + ψ·[log(w_t) - log(ρ_t) - log(u_t)] + ξ·log(g_t)
```
Where:
- `w_t - ρ_t - u_t` = excess demand pressure (wages relative to effective capacity)
- `g_t` = NY Fed GSCPI supply chain pressure index
- `ρ_t` = Nominal Potential GDP

### 3. New Exogenous Variables
| Variable | Data Source |
|----------|-------------|
| Capacity Utilization (`capu`) | FRED Total Capacity Utilization Index |
| Nominal Potential GDP (`ngdppot`) | FRED NGDPPOT |
| Supply Chain Pressure (`gscpi`) | NY Fed Global Supply Chain Pressure Index |

### 4. Inflation Expectations
**Unchanged** from original BB - same linear updating rules for short-run and long-run expectations.

### New Model Files
- `regression_new_model.py`: Python implementation of new model regressions
- `regression_new_model_pre_covid.py`: Pre-COVID sample estimation
- `plot_pred_v_actual_new_model.py`: Prediction vs actual plots for new model
- `cond_forecast_new_model.py`: Conditional forecasts with new model
- Output directories with `(New Model)` suffix contain results from the extended specification

## Software Requirements

- **Stata** (versions 17 or 18): For running regressions
- **MATLAB** (R2022a): For simulations, decompositions, and IRFs
  - Requires **Dynare 5.4** for calibrated model simulations
- **R** (version 4.3.1): For plotting and data visualization
- **Python** (version 3.8.8+): For principal component analysis

### Required Packages

**Python:**
- pandas
- sklearn (scikit-learn)
- os (built-in)

**R:**
- tidyverse
- magrittr
- ggplot2
- zoo
- tseries
- readxl
- grid
- lubridate
- reshape2

**Stata:**
- estout

## Repository Structure

The replication package is organized in a sequential workflow:

1. **(1) Data/**: Input data
   - `Public Data/`: FRED, Google Trends, and other public sources
     - `Regression_Data.dta` and `Regression_Data.xlsx`: Main regression data with data dictionary
     - `figure_6_data.xlsx`: Data for Figure 6
     - `SPF_data.xlsx`: Survey of Professional Forecasters expectations data
   - `Private Data/`: Bloomberg and Haver data (required only for PCA)
     - `Bloomberg_Commodity_Data.xlsx`: 19 commodity prices
     - `SP_GSCI.xlsx`: S&P Goldman Sachs Commodity Index

2. **(2) Regressions/**: Stata regression models
   - `regression_full.do`: Full sample SVAR estimation (1989 Q1 - 2023 Q2)
   - `regression_pre_covid.do`: Pre-COVID sample estimation (1989 Q1 - 2019 Q4 for most equations)
   - `Output Data/`: Contains regression coefficients, summary statistics, and simulation data
   - `Predicted vs. Actual/`: R code for out-of-sample prediction plots

3. **(3) Core Results/**: MATLAB simulations for main results
   - `Conditional Forecasts/`: Figure 14
   - `Decompositions/`: Figures 12 and 13
   - `IRFs/`: Figures 10 and 11

4. **(4) Other Results/**: Additional analyses
   - `PCA/`: Principal component analysis of commodity prices
   - `Simple Model/`: Calibrated DSGE model using Dynare
   - `Shortages/`: Figure 6 plots

5. **(5) Appendix/**: Alternative specifications and robustness checks
   - `PCE Regression/`: Alternative inflation measure
   - `SPF Regression/`: Alternative expectations data
   - `Shortage Alternatives/`: Alternative shortage measures
   - `V_U in the price equation/`: Alternative specifications
   - `Nonlinear V:U/`: Non-linearity tests

## Running the Code

The workflow must be executed sequentially:

### Step 1: Run Regressions (Stata)

```bash
cd "Replication Package/Code and Data/(2) Regressions"
# Run in Stata:
do regression_full.do
do regression_pre_covid.do
```

**Important:** Update working directory and export directory paths at the top of each .do file before running.

**Output:**
- `eq_coefficients_*.xlsx`: Estimated coefficients
- `summary_statistics_*.xlsx`: Summary statistics
- `eq_simulations_data_*.xls`: Data for simulations

### Step 2: Generate Prediction Plots (R)

```bash
cd "Replication Package/Code and Data/(2) Regressions/Predicted vs. Actual"
# Run in R:
Rscript plot_pred_v_actual.R
```

Note: Use output from pre-COVID sample for correct replication of Figures 3, 7, 8, and 9.

### Step 3: Core Results (MATLAB + R)

For each subfolder in `(3) Core Results/`:

**Conditional Forecasts (Figure 14):**
```matlab
cd "Replication Package/Code and Data/(3) Core Results/Conditional Forecasts"
% In MATLAB:
cond_forecast.m
```
Then plot with R: `Rscript plot_cf.R`

**Decompositions (Figures 12 and 13):**
```matlab
cd "Replication Package/Code and Data/(3) Core Results/Decompositions"
% In MATLAB:
decomp.m
```
Then plot with R: `Rscript plot_decomp.R`

**IRFs (Figures 10 and 11):**
```matlab
cd "Replication Package/Code and Data/(3) Core Results/IRFs"
% In MATLAB:
irf_simulations.m
```
Then plot with R: `Rscript plot_irfs.R`

**Important:** Always update working directory paths in both MATLAB and R scripts before running.

### Step 4: Other Results

**Principal Component Analysis:**
```bash
cd "Replication Package/Code and Data/(4) Other Results/PCA"
python pca.py
Rscript plot_pca.R
```

Note: Requires Bloomberg and Haver data to be refreshed in Excel files first.

**Simple Calibrated Model:**
```matlab
cd "Replication Package/Code and Data/(4) Other Results/Simple Model"
% In MATLAB:
Simple_Model.m
```

To run Dynare .mod files separately:
```matlab
dynare Simple_Eq_simulations_weak.mod
dynare Simple_Eq_simulations_strong.mod
```

**Shortage Plots (Figure 6):**
```bash
cd "Replication Package/Code and Data/(4) Other Results/Shortages"
Rscript plot_shortage.R
```

### Step 5: Appendix Results

Each subfolder in `(5) Appendix/` contains Stata .do files for alternative specifications. Run similarly to Step 1, updating file paths as needed.

## Architecture Notes

### Econometric Model Structure

The model is a Structural Vector Autoregression (SVAR) with four equations:
1. **Price equation**: CPI inflation
2. **Wage equation**: ECI wage growth (includes V/U ratio)
3. **Productivity equation**: Output per hour
4. **Expectations equation**: Inflation expectations

Key variables (Original BB):
- `gcpi`: CPI inflation (annualized quarterly)
- `gw`: Wage growth
- `gpty`: Productivity growth
- `grpe`, `grpf`: Relative prices (energy, food)
- `vu`: Vacancy-to-unemployment ratio (V/U)
- `cf1`, `cf10`: 1-year and 10-year inflation expectations
- `shortage`: Supply chain shortage measure (exogenous in BB)
- `diffcpicf`: Catch-up term (inflation minus lagged expectations)

Additional variables (New Model):
- `capu`: Capacity utilization (FRED Total Index)
- `ngdppot`: Nominal potential GDP (FRED NGDPPOT)
- `gscpi`: NY Fed Global Supply Chain Pressure Index
- `shortage`: Now endogenous, decomposed into excess demand and supply chain components

### COVID-19 Period Handling

- Pre-COVID sample: Most equations estimated through 2019 Q4, but price equation uses full sample
- Full sample: All equations through 2023 Q2
- Dummy variable `dummyq2_2020` for Q2 2020 shock
- Conditional forecast uses Q4 2019 as steady state for V/U, requiring wage equation constant adjustment (see `wage_constant.pdf`)

### Data Flow

1. Raw data → Stata regressions → Coefficient estimates
2. Coefficient estimates → MATLAB simulations → Forecast/decomposition results
3. MATLAB output → R scripts → Publication-quality figures

### File Path Patterns

All scripts require manual path updates:
- Stata: Look for "CHANGE PATH HERE" comments at top of .do files
- MATLAB: Update `cd` and data file paths in first few lines
- R: Update working directory with `setwd()` and Excel file paths
- Python: Update `os.chdir()` and file paths

## Common Tasks

**Reproduce main paper results:**
Run Steps 1-3 in sequence, using full sample regression output for Step 3.

**Reproduce out-of-sample predictions:**
Run Step 1 (pre-COVID regression), then Step 2.

**Add new commodity to PCA:**
Edit `pca.py` to include new Bloomberg/Haver ticker in data import and column renaming sections.

**Run alternative specification:**
Navigate to appropriate subfolder in `(5) Appendix/` and run the .do file after updating paths.

**Run Dynare models standalone:**
```matlab
cd "Replication Package/Code and Data/(4) Other Results/Simple Model/Code"
dynare Simple_Eq_simulations_weak.mod
dynare Simple_Eq_simulations_strong.mod
```

## Data Sources

- **FRED** (Federal Reserve Bank of St. Louis): Most macroeconomic data, including capacity utilization and nominal potential GDP
- **Google Trends**: Shortage search measure (exogenous in original BB)
- **NY Fed**: Global Supply Chain Pressure Index (GSCPI) - used in new model
- **Bloomberg**: Commodity futures prices (private, requires terminal access)
- **Haver Analytics**: S&P GSCI indices (private, requires subscription)
- **Survey of Professional Forecasters**: Inflation expectations (public)

## Estimated Runtime

Approximately 20 minutes on standard laptop for full replication.
