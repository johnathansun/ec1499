# Decomposition Methodology

## Overview

This document describes the mathematical methodology for decomposing pandemic-era inflation into contributions from different shocks. The approach uses **counterfactual simulations**: we compute what inflation (and other endogenous variables) would have been if a particular shock had not occurred, then attribute the difference to that shock.

---

## Model Structure

The modified Bernanke-Blanchard model consists of five equations estimated simultaneously:

### 1. Wage Equation

$$
gw_t = \alpha_0 + \sum_{i=1}^{4} \alpha_i \cdot gw_{t-i} + \sum_{i=1}^{4} \beta_i \cdot cf1_{t-i} + \gamma \cdot magpty_{t-1} + \sum_{i=1}^{4} \delta_i \cdot vu_{t-i} + \sum_{i=1}^{4} \phi_i \cdot diffcpicf_{t-i} + \sum_{i=1}^{4} \psi_i \cdot cu_{t-i} + \theta_1 \cdot D_{2020Q2} + \theta_2 \cdot D_{2020Q3}
$$

where:
- $gw_t$ = wage growth (ECI, annualized)
- $cf1_t$ = 1-year inflation expectations
- $magpty_t$ = productivity growth (moving average)
- $vu_t$ = vacancy-to-unemployment ratio
- $diffcpicf_t$ = catch-up term (realized inflation minus lagged expectations)
- $cu_t$ = detrended capacity utilization [NEW]
- $D_{2020Q2}, D_{2020Q3}$ = COVID dummy variables

### 2. Shortage Equation (Endogenous) [NEW]

$$
shortage_t = \alpha_0 + \sum_{i=1}^{4} \phi_i \cdot shortage_{t-i} + \sum_{i=0}^{4} \psi_i \cdot ed_{t-i} + \sum_{i=0}^{4} \xi_i \cdot gscpi_{t-i}
$$

where:
- $shortage_t$ = Google Trends shortage index
- $ed_t$ = excess demand (detrended)
- $gscpi_t$ = NY Fed Global Supply Chain Pressure Index

### 3. Price Equation

$$
gcpi_t = \alpha_0 + \beta \cdot magpty_t + \sum_{i=1}^{4} \gamma_i \cdot gcpi_{t-i} + \sum_{i=0}^{4} \delta_i \cdot gw_{t-i} + \sum_{i=0}^{4} \phi_i \cdot grpe_{t-i} + \sum_{i=0}^{4} \psi_i \cdot grpf_{t-i} + \sum_{i=0}^{4} \xi_i \cdot shortage_{t-i}
$$

where:
- $gcpi_t$ = CPI inflation (annualized)
- $grpe_t$ = relative price of energy (growth)
- $grpf_t$ = relative price of food (growth)

### 4. Long-Run Expectations Equation

$$
cf10_t = \sum_{i=1}^{4} \alpha_i \cdot cf10_{t-i} + \sum_{i=0}^{4} \beta_i \cdot gcpi_{t-i}
$$

### 5. Short-Run Expectations Equation

$$
cf1_t = \sum_{i=1}^{4} \alpha_i \cdot cf1_{t-i} + \sum_{i=0}^{4} \beta_i \cdot cf10_{t-i} + \sum_{i=0}^{4} \gamma_i \cdot gcpi_{t-i}
$$

### Catch-Up Term (Identity)

$$
diffcpicf_t = \frac{1}{4}(gcpi_t + gcpi_{t-1} + gcpi_{t-2} + gcpi_{t-3}) - cf1_{t-4}
$$

---

## Counterfactual Decomposition Method

### Basic Principle

For each shock $x$, we compute:

$$
\text{Contribution of } x \text{ to } y = y^{baseline}_t - y^{remove\_x}_t
$$

where:
- $y^{baseline}_t$ = simulated value with all shocks active
- $y^{remove\_x}_t$ = simulated value with shock $x$ set to its steady-state value

### Key Innovation: Endogenous Excess Demand

**Important**: Excess demand is computed **endogenously** from simulated wage levels in both baseline and counterfactual simulations. This captures the full feedback loop:

$$
\text{shock} \to gw \to W \to ed \to shortage \to gcpi
$$

When a shock (e.g., V/U) reduces wage growth:
1. Lower wage growth → lower wage level accumulation
2. Lower wage level → lower excess demand
3. Lower excess demand → lower shortages (via shortage equation)
4. Lower shortages → lower inflation

This ensures the decomposition captures **indirect effects** through the wage → excess demand → shortage channel.

### Simulation Procedure

For each time period $t = 4, 5, \ldots, T$:

**Step 1: Compute Wages**
$$
gw_t = f(gw_{t-1:t-4}, cf1_{t-1:t-4}, magpty_{t-1}, vu_{t-1:t-4}, diffcpicf_{t-1:t-4}, cu_{t-1:t-4}, D_t)
$$

**Step 2: Update Wage Level and Compute Excess Demand** (endogenous)
$$
\ln(W_t) = \ln(W_{t-1}) + \frac{gw_t}{400}
$$
$$
\ln(NGDPPOT_t) = \ln(NGDPPOT_{t-1}) + \frac{g_{NGDPPOT}}{400}
$$
$$
ed_t^{raw} = \ln(W_t) - \ln(NGDPPOT_t) - \ln(TCU_t/100)
$$
$$
ed_t = ed_t^{raw} - \overline{ed}^{40q}_t
$$

**Step 3: Compute Shortage** (uses endogenous excess demand)
$$
shortage_t = f(shortage_{t-1:t-4}, ed_{t:t-4}, gscpi_{t:t-4})
$$

**Step 4: Compute Prices**
$$
gcpi_t = f(magpty_t, gcpi_{t-1:t-4}, gw_{t:t-4}, grpe_{t:t-4}, grpf_{t:t-4}, shortage_{t:t-4})
$$

**Step 5: Update Catch-Up Term**
$$
diffcpicf_t = \frac{1}{4}\sum_{i=0}^{3} gcpi_{t-i} - cf1_{t-4}
$$

**Step 6: Update Expectations**
$$
cf10_t = f(cf10_{t-1:t-4}, gcpi_{t:t-4})
$$
$$
cf1_t = f(cf1_{t-1:t-4}, cf10_{t:t-4}, gcpi_{t:t-4})
$$

### Initial Conditions

The first 4 periods use historical data as initial conditions. The dynamic simulation begins at $t = 4$.

---

## Shocks Analyzed

### Original Bernanke-Blanchard Shocks

| Shock | Counterfactual Setting | Steady State Value |
|-------|------------------------|-------------------|
| Energy prices ($grpe$) | Set to 0 | 0 (no relative price change) |
| Food prices ($grpf$) | Set to 0 | 0 (no relative price change) |
| V/U ratio ($vu$) | Set to initial value | $vu^* = vu_{t=4}$ |
| Shortages ($shortage$) | Set to steady state | $shortage^* = 5.0$ |
| Productivity ($magpty$) | Set to long-run average | $magpty^* = 2.0$ |
| Q2 2020 dummy | Set to 0 | 0 |
| Q3 2020 dummy | Set to 0 | 0 |

### New Model Shocks

| Shock | Counterfactual Setting | Steady State Value |
|-------|------------------------|-------------------|
| Excess demand ($ed$) | Set to initial value | $ed^* = ed_{t=4}$ |
| GSCPI ($gscpi$) | Set to 0 | 0 (no supply chain pressure) |
| Capacity utilization ($cu$) | Set to 0 | 0 (on trend) |

---

## Contribution Calculation

### Direct Contribution

For a shock $x$ removed in simulation $s$:

$$
x\_contr\_gcpi_t = gcpi^{baseline}_t - gcpi^{s}_t
$$

$$
x\_contr\_gw_t = gw^{baseline}_t - gw^{s}_t
$$

$$
x\_contr\_shortage_t = shortage^{baseline}_t - shortage^{s}_t
$$

### Interpretation

- **Positive contribution**: The shock pushed the variable above what it would have been otherwise
- **Negative contribution**: The shock pushed the variable below what it would have been otherwise

---

## Excess Demand Component Attribution

### Excess Demand Definition

Excess demand measures the gap between wages and effective capacity:

$$
ed_t = \ln(W_t) - \ln(NGDPPOT_t) - \ln(TCU_t/100) - \overline{ed}^{40q}_t
$$

where:
- $W_t$ = wage level (ECI index)
- $NGDPPOT_t$ = nominal potential GDP
- $TCU_t$ = total capacity utilization (percent)
- $\overline{ed}^{40q}_t$ = 40-quarter rolling mean (trend)

### Component Decomposition

The deviation of excess demand from a reference period can be decomposed:

$$
\Delta ed_t = \underbrace{\Delta \ln(W_t)}_{\text{wage component}} - \underbrace{\Delta \ln(NGDPPOT_t)}_{\text{potential GDP component}} - \underbrace{\Delta \ln(TCU_t/100)}_{\text{capacity util component}}
$$

### Proportional Attribution

Let:
- $w\_contr\_ed_t = \ln(W_t) - \ln(W_{ref})$ (wage contribution to ED)
- $ngdppot\_contr\_ed_t = -[\ln(NGDPPOT_t) - \ln(NGDPPOT_{ref})]$ (potential GDP contribution)
- $cu\_contr\_ed_t = -[\ln(TCU_t/100) - \ln(TCU_{ref}/100)]$ (capacity util contribution)

The share of each component in total ED deviation:

$$
share^{wage}_t = \frac{w\_contr\_ed_t}{\Delta ed_t}
$$

$$
share^{cu}_t = \frac{cu\_contr\_ed_t}{\Delta ed_t}
$$

$$
share^{ngdppot}_t = \frac{ngdppot\_contr\_ed_t}{\Delta ed_t}
$$

### Attributed Contributions

The contribution of each ED component to shortages and inflation:

$$
wage\_contr\_shortage_t = ed\_contr\_shortage_t \times share^{wage}_t
$$

$$
wage\_contr\_gcpi\_via\_ed_t = ed\_contr\_gcpi_t \times share^{wage}_t
$$

(Similarly for capacity utilization and potential GDP components)

---

## Capacity Utilization: Total Effect on Inflation

Capacity utilization affects inflation through two channels:

### 1. Direct Channel (via Wages)

$$
cu \xrightarrow{\text{wage eq.}} gw \xrightarrow{\text{price eq.}} gcpi
$$

This is captured by: $cu\_contr\_gcpi\_direct = gcpi^{baseline} - gcpi^{remove\_cu}$

### 2. Indirect Channel (via Excess Demand → Shortages)

$$
cu \xrightarrow{ed = \ln W - \ln NGDPPOT - \ln TCU} ed \xrightarrow{\text{shortage eq.}} shortage \xrightarrow{\text{price eq.}} gcpi
$$

This is captured by: $cu\_contr\_gcpi\_via\_ed = ed\_contr\_gcpi \times share^{cu}$

### Total Effect

$$
cu\_total\_contr\_gcpi_t = cu\_contr\_gcpi\_direct_t + cu\_contr\_gcpi\_via\_ed_t
$$

---

## Key Insight: Shortage Attribution

The modified model allows us to decompose the "shortage" contribution to inflation into:

1. **Demand-driven shortages**: Excess demand from wages outpacing capacity
2. **Supply-chain-driven shortages**: Global supply chain pressure (GSCPI)

This answers the question: *How much of shortage-driven inflation was caused by demand exceeding supply capacity vs. supply chain disruptions?*

---

## Output Files

| File | Description |
|------|-------------|
| `baseline.xlsx` | Baseline simulation with all shocks active |
| `remove_grpe.xlsx` | Counterfactual: no energy price shocks |
| `remove_grpf.xlsx` | Counterfactual: no food price shocks |
| `remove_vu.xlsx` | Counterfactual: V/U at steady state |
| `remove_shortage.xlsx` | Counterfactual: shortage at steady state |
| `remove_magpty.xlsx` | Counterfactual: productivity at long-run average |
| `remove_2020q2.xlsx` | Counterfactual: no Q2 2020 dummy |
| `remove_2020q3.xlsx` | Counterfactual: no Q3 2020 dummy |
| `remove_excess_demand.xlsx` | Counterfactual: excess demand at steady state [NEW] |
| `remove_gscpi.xlsx` | Counterfactual: no supply chain pressure [NEW] |
| `remove_cu.xlsx` | Counterfactual: capacity utilization on trend [NEW] |
| `remove_all.xlsx` | Counterfactual: all shocks removed |
| `excess_demand_components.xlsx` | Component-level attribution [NEW] |

---

## References

- Bernanke, B. S., & Blanchard, O. J. (2023). What Caused the U.S. Pandemic-Era Inflation? *Brookings Papers on Economic Activity*.
- Liang & Sun (2025). Extended model with endogenous shortages.
