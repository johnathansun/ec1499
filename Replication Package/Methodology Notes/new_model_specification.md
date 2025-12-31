# New Model Regression Specification

### Other Notes

1. Original BB paper says productivity growth is lagged ($gpty_{t-1}$) in both wage and price equation. Looking at the Stata, the contemporaneous version is used in the price equation, while the lagged version is used in the wage equation
2. Detrended series: capacity utilization and excess demand, both by subtracting rolling 10Y mean

   1. Capacity utilization has decreased over time
   2. Excess demand is decreasing since potential GDP increasing faster than wages. Might be plausible since labor share of income has fallen since the 80s; productivity growing faster than real wages for many workers; globalization, weaker unions, all this kind of stuff shifts income from wages to profits.
3. 

### New reg spec

This document describes the regressions estimated in `regression_new_model_merged.py`.

## Equations

### 1. Wage Equation

$$
gw_t = \alpha_0 + \sum_{i=1}^{4} \alpha_i \cdot gw_{t-i} + \sum_{i=1}^{4} \beta_i \cdot cf1_{t-i} + \gamma \cdot magpty_{t-1} + \sum_{i=1}^{4} \delta_i \cdot vu_{t-i} + \sum_{i=1}^{4} \phi_i \cdot diffcpicf_{t-i} + \sum_{i=1}^{4} \psi_i \cdot cu_{t-i} + \epsilon_t
$$

**Constraint:** $\sum_{i=1}^{4} \alpha_i + \sum_{i=1}^{4} \beta_i = 1$

**Sample:** Pre-COVID (≤2019 Q4) or full sample depending on configuration. Full sample includes Q2 and Q3 2020 dummy variables.

**Modification from BB:** Added detrended capacity utilization level terms ($cu_{t-i}$).

---

### 2. Shortage Equation (New)

$$
shortage_t = \alpha_0 + \sum_{i=1}^{4} \phi_i \cdot shortage_{t-i} + \sum_{i=0}^{4} \psi_i \cdot excessdemand_{t-i} + \sum_{i=0}^{4} \xi_i \cdot gscpi_{t-i} + \epsilon_t
$$

**Constraint:** None (unconstrained OLS)

**Sample:** Always full sample

**New equation:** Endogenizes shortages as a function of lagged shortages, excess demand, and supply chain pressure.

---

### 3. Price Equation

$$
gcpi_t = \alpha_0 + \beta \cdot magpty_t + \sum_{i=1}^{4} \gamma_i \cdot gcpi_{t-i} + \sum_{i=0}^{4} \delta_i \cdot gw_{t-i} + \sum_{i=0}^{4} \phi_i \cdot grpe_{t-i} + \sum_{i=0}^{4} \psi_i \cdot grpf_{t-i} + \sum_{i=0}^{4} \xi_i \cdot shortage_{t-i} + \epsilon_t
$$

**Constraint:** $\sum_{i=1}^{4} \gamma_i + \sum_{i=0}^{4} \delta_i = 1$

**Sample:** Always full sample

**Same as BB.**

---

### 4. Short-Run Expectations Equation

$$
cf1_t = \sum_{i=1}^{4} \alpha_i \cdot cf1_{t-i} + \sum_{i=0}^{4} \beta_i \cdot cf10_{t-i} + \sum_{i=0}^{4} \gamma_i \cdot gcpi_{t-i}
$$

**Constraint:** $\sum_{i=1}^{4} \alpha_i + \sum_{i=0}^{4} \beta_i + \sum_{i=0}^{4} \gamma_i = 1$ (no constant)

**Sample:** Pre-COVID or full sample depending on configuration

**Same as BB.**

---

### 5. Long-Run Expectations Equation

$$
cf10_t = \sum_{i=1}^{4} \alpha_i \cdot cf10_{t-i} + \sum_{i=0}^{4} \beta_i \cdot gcpi_{t-i}
$$

**Constraint:** $\sum_{i=1}^{4} \alpha_i + \sum_{i=0}^{4} \beta_i = 1$ (no constant)

**Sample:** Pre-COVID or full sample depending on configuration

**Same as BB.**

---

## Variable Definitions and Units

| Variable           | Definition                                                                                                                                                                                             | Units                               | Source                |
| ------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ----------------------------------- | --------------------- |
| $gcpi_t$         | $400 \times [\ln(CPI_t) - \ln(CPI_{t-1})]$                                                                                                                                                           | Percent per year (annualized)       | FRED CPIAUCSL         |
| $gw_t$           | $400 \times [\ln(ECI_t) - \ln(ECI_{t-1})]$                                                                                                                                                           | Percent per year (annualized)       | FRED ECIWAG           |
| $gpty_t$         | $400 \times [\ln(OPHNFB_t) - \ln(OPHNFB_{t-1})]$                                                                                                                                                     | Percent per year (annualized)       | FRED OPHNFB           |
| $magpty_t$       | $\frac{1}{8} \sum_{i=0}^{7} gpty_{t-i}$                                                                                                                                                              | Percent per year                    | Derived               |
| $grpe_t$         | $400 \times [\ln(rpe_t) - \ln(rpe_{t-1})]$ where $rpe = CPIENGSL / ECIWAG$                                                                                                                         | Percent per year (annualized)       | Derived               |
| $grpf_t$         | $400 \times [\ln(rpf_t) - \ln(rpf_{t-1})]$ where $rpf = CPIUFDSL / ECIWAG$                                                                                                                         | Percent per year (annualized)       | Derived               |
| $vu_t$           | Vacancy-to-unemployment ratio                                                                                                                                                                          | Ratio (unitless)                    | FRED VOVERU           |
| $cf1_t$          | 1-year ahead inflation expectations                                                                                                                                                                    | Percent                             | Cleveland Fed         |
| $cf10_t$         | 10-year ahead inflation expectations                                                                                                                                                                   | Percent                             | Cleveland Fed         |
| $shortage_t$     | Google Trends "shortage" search index (missing filled with 5)                                                                                                                                          | Index (unitless)                    | Google Trends         |
| $diffcpicf_t$    | $\frac{1}{4} \sum_{i=0}^{3} gcpi_{t-i} - cf1_{t-4}$                                                                                                                                                  | Percent                             | Derived               |
| $cu_t$           | $\ln(TCU_t/100) - \ln(\bar{TCU}_t^{10yr}/100)$ where $\bar{TCU}_t^{10yr}$ is 10-year rolling mean of TCU                                                                                           | Log deviation from trend (unitless) | Derived from FRED TCU |
| $excessdemand_t$ | Standardized and detrended: $\frac{x_t - \bar{x}}{\sigma_x}$ where $x_t = [\ln(ECIWAG_t) - \ln(NGDPPOT_t) - \ln(TCU_t/100)] - \bar{x}_t^{40q}$ and $\bar{x}_t^{40q}$ is a 40-quarter rolling mean | Standard deviations from mean       | Derived               |
| $gscpi_t$        | Global Supply Chain Pressure Index                                                                                                                                                                     | Standard deviations from mean       | NY Fed                |

---

## Notes

1. **TCU usage:** Capacity utilization is detrended in-script as $cu_t = \ln(TCU_t/100) - \ln(\bar{TCU}_t^{10yr}/100)$. The detrended $cu_t$ enters the wage equation (as lags). For excess demand, we use raw $\ln(TCU_t/100)$ (not detrended), then detrend and standardize the final excess demand variable.
2. **Sample periods:**

   - `USE_PRE_COVID_SAMPLE = False`: All equations use full sample (1989 Q1 - 2023 Q2)
   - `USE_PRE_COVID_SAMPLE = True`: Wage and expectations equations use pre-COVID sample (≤2019 Q4); shortage and price equations use full sample
3. **Annualization:** Growth rates are annualized by multiplying quarterly log-differences by 400.

---

## Model Specifications and Configuration Flags

The regression scripts support multiple model specifications via configuration flags:

### Configuration Flags

| Flag | Description | Default |
|------|-------------|---------|
| `USE_PRE_COVID_SAMPLE` | If True, wage and expectations equations estimated on pre-COVID sample (≤2019 Q4) | True |
| `USE_LOG_CU_WAGES` | If True, use log-detrended CU; if False, use level-detrended CU | False |
| `USE_CONTEMP_CU` | If True, include contemporaneous CU in wage equation (L0-L4); if False, only lags (L1-L4) | True |
| `USE_DETRENDED_EXCESS_DEMAND` | If True, detrend excess demand by 40-quarter rolling mean | False |

### Model Specification Summary

| Specification | CU in Wages | Contemp CU | Log CU | Detrend ED | Description |
|---------------|-------------|------------|--------|------------|-------------|
| BB Original | No | - | - | - | Original Bernanke-Blanchard (no CU, exogenous shortage) |
| New Base | Yes (L1-L4) | No | No | No | Adds CU to wages, endogenous shortage |
| New +CU(0) | Yes (L0-L4) | Yes | No | No | Adds contemporaneous CU |
| New +LogCU | Yes (L1-L4) | No | Yes | No | Uses log(CU) instead of level |
| New +Log+CU(0) | Yes (L0-L4) | Yes | Yes | No | Log CU with contemporaneous |
| New +DetrendED | Yes (L1-L4) | No | No | Yes | Detrends excess demand |

---

## Output Directories

Output files are automatically named based on configuration:

| Directory | Configuration |
|-----------|---------------|
| `Output Data (New)` | Full sample, base specification |
| `Output Data (New Pre Covid)` | Pre-COVID sample, base specification |
| `Output Data (New Pre Covid Contemp CU)` | Pre-COVID, contemporaneous CU |
| `Output Data (New Pre Covid Log CU)` | Pre-COVID, log CU |
| `Output Data (New Pre Covid Log CU Contemp CU)` | Pre-COVID, log CU + contemporaneous |
| `Output Data (New Pre Covid Detrended ED)` | Pre-COVID, detrended excess demand |
| `Old Output/Output Data (Pre Covid Sample)` | Original BB model |

Each directory contains:
- `eq_coefficients_*.xlsx` - Estimated coefficients for each equation
- `eq_simulations_data_*.xlsx` - Data with fitted values and residuals
- `summary_stats_*.xlsx` - Summary statistics (transposed format)

---

## Scripts

### Main Regression Scripts

| Script | Description |
|--------|-------------|
| `regression_new_model_refactored.py` | **Main script.** Declarative variable definitions, auto-generated summaries. Configure via flags at top. |
| `regression_new_model_merged.py` | Original merged script (deprecated, use refactored version) |

### Analysis Scripts

| Script | Description |
|--------|-------------|
| `compare_model_specifications.py` | Compares out-of-sample RMSE across all model specifications |
| `plot_excess_demand.py` | Plots excess demand time series |

### Predicted vs. Actual

| Script | Location |
|--------|----------|
| `plot_pred_v_actual_new_model.py` | `Predicted vs. Actual/` - Plots actual vs predicted for each variable |

---

## Out-of-Sample Performance Comparison

RMSE for out-of-sample predictions (2020 Q1 onwards), estimated on pre-COVID sample:

### Wage Growth Prediction

| Model | RMSE | % vs BB |
|-------|------|---------|
| BB Original | 1.202 | 0% |
| New Base | 1.452 | +21% |
| New +CU(0) | 1.238 | +3% |
| New +LogCU | 1.455 | +21% |
| New +Log+CU(0) | 1.237 | +3% |
| New +DetrendED | 1.452 | +21% |

### Key Findings

1. **BB Original has best wage prediction** - The original model without CU predicts wages best out-of-sample
2. **Adding lagged CU hurts prediction** - New Base is 21% worse than BB Original
3. **Contemporaneous CU recovers performance** - Adding CU(0) brings it to only 3% worse
4. **Detrending ED doesn't affect wages** - ED only enters shortage equation, not wage equation
5. **Inflation/expectations identical** - These equations are the same across New model variants

### Interpretation

Adding capacity utilization to the wage equation may overfit to the pre-COVID sample. The CU coefficients capture patterns that don't generalize to the COVID/post-COVID period. However, contemporaneous CU provides timelier information that partially compensates.

---

## Variable Construction Details

### Capacity Utilization (cu)

**Level-detrended (default):**
$$cu_t = \frac{TCU_t - \bar{TCU}_t^{40q}}{100}$$

**Log-detrended (USE_LOG_CU_WAGES=True):**
$$cu_t = \ln(TCU_t/100) - \ln(\bar{TCU}_t^{40q}/100)$$

### Excess Demand

**Raw excess demand:**
$$ed_t^{raw} = \ln(W_t) - \ln(NGDPPOT_t) - \ln(TCU_t/100)$$

**Detrended (USE_DETRENDED_EXCESS_DEMAND=True):**
$$ed_t = ed_t^{raw} - \bar{ed}_t^{40q}$$

where $\bar{ed}_t^{40q}$ is the 40-quarter rolling mean.

**Non-detrended (default):**
$$ed_t = ed_t^{raw}$$
