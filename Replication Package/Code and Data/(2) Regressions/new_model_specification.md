# New Model Regression Specification

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

| Variable           | Definition                                                                     | Units                                   | Source              |
| ------------------ | ------------------------------------------------------------------------------ | --------------------------------------- | ------------------- |
| $gcpi_t$         | $400 \times [\ln(CPI_t) - \ln(CPI_{t-1})]$                                   | Percent per year (annualized)           | FRED CPIAUCSL       |
| $gw_t$           | $400 \times [\ln(ECI_t) - \ln(ECI_{t-1})]$                                   | Percent per year (annualized)           | FRED ECIWAG         |
| $gpty_t$         | $400 \times [\ln(OPHNFB_t) - \ln(OPHNFB_{t-1})]$                             | Percent per year (annualized)           | FRED OPHNFB         |
| $magpty_t$       | $\frac{1}{8} \sum_{i=0}^{7} gpty_{t-i}$                                      | Percent per year                        | Derived             |
| $grpe_t$         | $400 \times [\ln(rpe_t) - \ln(rpe_{t-1})]$ where $rpe = CPIENGSL / ECIWAG$ | Percent per year (annualized)           | Derived             |
| $grpf_t$         | $400 \times [\ln(rpf_t) - \ln(rpf_{t-1})]$ where $rpf = CPIUFDSL / ECIWAG$ | Percent per year (annualized)           | Derived             |
| $vu_t$           | Vacancy-to-unemployment ratio                                                  | Ratio (unitless)                        | FRED VOVERU         |
| $cf1_t$          | 1-year ahead inflation expectations                                            | Percent                                 | Cleveland Fed       |
| $cf10_t$         | 10-year ahead inflation expectations                                           | Percent                                 | Cleveland Fed       |
| $shortage_t$     | Google Trends "shortage" search index (missing filled with 5)                  | Index (unitless)                        | Google Trends       |
| $diffcpicf_t$    | $\frac{1}{4} \sum_{i=0}^{3} gcpi_{t-i} - cf1_{t-4}$                          | Percent                                 | Derived             |
| $cu_t$           | $\ln(TCU_t) - \ln(\bar{TCU}_t^{10yr})$ where $\bar{TCU}_t^{10yr}$ is 10-year rolling mean | Log deviation from trend (unitless) | Derived from FRED TCU |
| $excessdemand_t$ | Standardized and detrended: $\frac{x_t - \bar{x}}{\sigma_x}$ where $x_t = [\ln(ECIWAG_t) - \ln(NGDPPOT_t) - cu_t] - \bar{x}_t^{40q}$ and $\bar{x}_t^{40q}$ is a 40-quarter rolling mean | Standard deviations from mean | Derived             |
| $gscpi_t$        | Global Supply Chain Pressure Index                                             | Standard deviations from mean           | NY Fed              |

---

## Notes

1. **TCU usage:** Capacity utilization is detrended in-script as $cu_t = \ln(TCU_t) - \ln(\bar{TCU}_t^{10yr})$. This detrended level enters both the wage equation (as lags) and the shortage equation (via excess demand).
2. **Sample periods:**

   - `USE_PRE_COVID_SAMPLE = False`: All equations use full sample (1989 Q1 - 2023 Q2)
   - `USE_PRE_COVID_SAMPLE = True`: Wage and expectations equations use pre-COVID sample (≤2019 Q4); shortage and price equations use full sample
3. **Annualization:** Growth rates are annualized by multiplying quarterly log-differences by 400.
