# Conditional Forecast Methodology

## Overview

This document describes the conditional forecast methodology for the modified Bernanke-Blanchard model with endogenous shortages. The key innovation is that **shortages are now endogenous**, determined by excess demand (which itself depends on wages) and supply chain pressure.

---

## Model Equations

### 1. Wage Equation

$$
gw_t = \alpha_0^{adj} + \sum_{i=1}^{4} \alpha_i \cdot gw_{t-i} + \sum_{i=1}^{4} \beta_i \cdot cf1_{t-i} + \gamma \cdot magpty_{t-1} + \sum_{i=1}^{4} \delta_i \cdot vu_{t-i} + \sum_{i=1}^{4} \phi_i \cdot diffcpicf_{t-i} + \sum_{i=1}^{4} \psi_i \cdot cu_{t-i}
$$

where $\alpha_0^{adj}$ is the adjusted wage constant (see Section 4).

### 2. Excess Demand (Endogenous)

$$
ed_t^{raw} = \ln(W_t) - \ln(NGDPPOT_t) - \ln(TCU_t/100)
$$

$$
ed_t = ed_t^{raw} - \bar{ed}_t^{40q}
$$

where:
- $W_t$ is the wage level (ECI index)
- $NGDPPOT_t$ is nominal potential GDP
- $TCU_t$ is capacity utilization (percent)
- $\bar{ed}_t^{40q}$ is the 40-quarter rolling mean of raw excess demand

The wage level evolves as:
$$
\ln(W_t) = \ln(W_{t-1}) + \frac{gw_t}{400}
$$

### 3. Shortage Equation (Endogenous)

$$
shortage_t = \alpha_0 + \sum_{i=1}^{4} \phi_i \cdot shortage_{t-i} + \sum_{i=0}^{4} \psi_i \cdot ed_{t-i} + \sum_{i=0}^{4} \xi_i \cdot gscpi_{t-i}
$$

### 4. Price Equation

$$
gcpi_t = \alpha_0 + \beta \cdot magpty_t + \sum_{i=1}^{4} \gamma_i \cdot gcpi_{t-i} + \sum_{i=0}^{4} \delta_i \cdot gw_{t-i} + \sum_{i=0}^{4} \phi_i \cdot grpe_{t-i} + \sum_{i=0}^{4} \psi_i \cdot grpf_{t-i} + \sum_{i=0}^{4} \xi_i \cdot shortage_{t-i}
$$

### 5. Expectations Equations

**Short-run (1-year):**
$$
cf1_t = \sum_{i=1}^{4} \alpha_i \cdot cf1_{t-i} + \sum_{i=0}^{4} \beta_i \cdot cf10_{t-i} + \sum_{i=0}^{4} \gamma_i \cdot gcpi_{t-i}
$$

**Long-run (10-year):**
$$
cf10_t = \sum_{i=1}^{4} \alpha_i \cdot cf10_{t-i} + \sum_{i=0}^{4} \beta_i \cdot gcpi_{t-i}
$$

---

## Variable Classification

| Variable | Type | Steady State Value |
|----------|------|-------------------|
| $gw_t$ | Endogenous | $\pi^* + magpty^*$ |
| $gcpi_t$ | Endogenous | $\pi^*$ (target inflation) |
| $cf1_t$ | Endogenous | $\pi^*$ |
| $cf10_t$ | Endogenous | $\pi^*$ |
| $shortage_t$ | Endogenous | $shortage^*$ (implied) |
| $ed_t$ | Endogenous | 0 (on trend) |
| $vu_t$ | Exogenous | $vu^* = 1.2$ |
| $grpe_t$ | Exogenous | 0 |
| $grpf_t$ | Exogenous | 0 |
| $magpty_t$ | Exogenous | $magpty^* = 1.0$ |
| $cu_t$ | Exogenous | $cu^* = 0$ (on trend) |
| $gscpi_t$ | Exogenous | $gscpi^* = 0$ |
| $g_{NGDPPOT}$ | Exogenous | 4% (annualized) |
| $TCU^*$ | Exogenous | 75% |

---

## Simulation Loop

The simulation proceeds sequentially for $t = 4, 5, \ldots, T$:

### Step 1: Compute Wages
Wages depend only on lagged values, so they can be computed first:
$$
gw_t = f(gw_{t-1:t-4}, cf1_{t-1:t-4}, magpty_{t-1}, vu_{t-1:t-4}, diffcpicf_{t-1:t-4}, cu_{t-1:t-4})
$$

### Step 2: Update Levels and Compute Excess Demand
Update the wage level:
$$
\ln(W_t) = \ln(W_{t-1}) + \frac{gw_t}{400}
$$

Update potential GDP level (exogenous):
$$
\ln(NGDPPOT_t) = \ln(NGDPPOT_{t-1}) + \frac{g_{NGDPPOT}}{400}
$$

Compute raw excess demand:
$$
ed_t^{raw} = \ln(W_t) - \ln(NGDPPOT_t) - \ln(TCU^*/100)
$$

Compute rolling trend and detrend:
$$
\bar{ed}_t^{40q} = \frac{1}{\min(40, t+1)} \sum_{i=0}^{\min(39,t)} ed_{t-i}^{raw}
$$
$$
ed_t = ed_t^{raw} - \bar{ed}_t^{40q}
$$

### Step 3: Compute Shortage
Using the endogenous excess demand:
$$
shortage_t = f(shortage_{t-1:t-4}, ed_{t:t-4}, gscpi_{t:t-4})
$$

### Step 4: Compute Prices
Using the endogenous shortage:
$$
gcpi_t = f(magpty_t, gcpi_{t-1:t-4}, gw_{t:t-4}, grpe_{t:t-4}, grpf_{t:t-4}, shortage_{t:t-4})
$$

### Step 5: Update Catch-up Term
$$
diffcpicf_t = \frac{1}{4}(gcpi_t + gcpi_{t-1} + gcpi_{t-2} + gcpi_{t-3}) - cf1_{t-4}
$$

### Step 6: Compute Expectations
Long-run first (since short-run depends on it):
$$
cf10_t = f(cf10_{t-1:t-4}, gcpi_{t:t-4})
$$
$$
cf1_t = f(cf1_{t-1:t-4}, cf10_{t:t-4}, gcpi_{t:t-4})
$$

---

## Wage Constant Adjustment

The wage equation constant is adjusted to ensure the model converges to a desired steady state. Following Bernanke-Blanchard, we target:

- $vu^* = 1.2$ (target vacancy-unemployment ratio)
- $cu^* = 0$ (capacity utilization on trend)
- $magpty^* = 1.0$ (long-run productivity growth)
- $gscpi^* = 0$ (no supply chain pressure)
- $ed^* = 0$ (excess demand on trend)

The adjusted constant is:
$$
\alpha_0^{adj} = -\left(\sum_{i=1}^4 \delta_i\right) \cdot vu^* - \left(\sum_{i=1}^4 \psi_i\right) \cdot cu^* - (1 - \sum_{i=1}^4 \alpha_i) \cdot \left[ \frac{\beta_{magpty}^{gcpi}}{1 - \sum \gamma^{gcpi}} + \frac{\gamma^{gw}}{1 - \sum \alpha^{gw}} \right] \cdot magpty^* - (1 - \sum_{i=1}^4 \alpha_i) \cdot \frac{\sum \xi^{gcpi}}{1 - \sum \gamma^{gcpi}} \cdot shortage^* - (1 - \sum_{i=1}^4 \alpha_i) \cdot \frac{\alpha_0^{gcpi}}{1 - \sum \gamma^{gcpi}}
$$

where $shortage^*$ is the implied steady-state shortage from the shortage equation:
$$
shortage^* = \frac{\alpha_0^{shortage} + \left(\sum \psi\right) \cdot ed^* + \left(\sum \xi\right) \cdot gscpi^*}{1 - \sum \phi}
$$

---

## Steady State Dynamics of Excess Demand

In steady state with constant growth rates:
- Wage growth: $gw^*$
- NGDPPOT growth: $g_{NGDPPOT} = 4\%$

The raw excess demand grows at rate:
$$
\frac{d(ed^{raw})}{dt} = \frac{gw^* - g_{NGDPPOT}}{400}
$$

With the rolling mean trend, the detrended excess demand converges to:
$$
ed^* = \frac{40-1}{2} \cdot \frac{gw^* - g_{NGDPPOT}}{400} = 19.5 \cdot \frac{gw^* - g_{NGDPPOT}}{400}
$$

For $gw^* \approx 3\%$ (2% inflation + 1% productivity) and $g_{NGDPPOT} = 4\%$:
$$
ed^* \approx 19.5 \cdot \frac{-1}{400} \approx -0.049
$$

This small negative value reflects wages growing slower than nominal potential GDP, consistent with a declining labor share of income.

---

## V/U Path Scenarios

The conditional forecast examines three scenarios for the vacancy-unemployment ratio:

1. **Low ($vu \to 0.8$)**: Labor market loosens significantly
2. **Target ($vu \to 1.2$)**: Labor market returns to pre-pandemic normal
3. **High ($vu \to 1.8$)**: Labor market remains tight

The transition occurs linearly over 8 quarters:
$$
vu_t = vu_{t-1} + \frac{vu^{target} - vu_0}{8}
$$

---

## Key Differences from Original Bernanke-Blanchard

1. **Endogenous Shortages**: Shortages are no longer exogenous but depend on excess demand and supply chain pressure (GSCPI)

2. **Capacity Utilization in Wages**: The wage equation includes detrended capacity utilization as an additional driver

3. **Excess Demand Feedback**: Wages affect excess demand, which affects shortages, which affects prices, creating an additional feedback channel:

$$
gw_t \to W_t \to ed_t \to shortage_t \to gcpi_t \to cf1_t \to gw_{t+1}
$$

4. **Dynamic Trend**: The excess demand trend evolves as a rolling mean rather than being fixed, ensuring proper convergence in steady state

---

## Output Files

The forecast generates three Excel files:
- `terminal_low.xlsx`: $vu \to 0.8$
- `terminal_mid.xlsx`: $vu \to 1.2$ (target steady state)
- `terminal_high.xlsx`: $vu \to 1.8$

Each contains time series of all simulated variables including the endogenous shortage and excess demand paths.
