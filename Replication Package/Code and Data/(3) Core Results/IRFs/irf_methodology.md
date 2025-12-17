# Impulse Response Function Methodology

## Overview

This document describes the methodology for computing impulse response functions (IRFs) in the modified Bernanke-Blanchard model with endogenous shortages. IRFs measure how the system responds to a one-time shock to an exogenous variable, tracing out the dynamic propagation through the model.

The key innovation is that **excess demand is endogenous**: wages are computed first, which determine the wage level, which then feeds into excess demand, which affects shortages, which finally affect prices.

---

## What is an Impulse Response Function?

An IRF shows the time path of an endogenous variable (e.g., inflation) following a one-time shock to an exogenous variable (e.g., energy prices). Starting from a steady state where all variables equal zero (deviations from steady state), we introduce a shock and simulate the model forward to see how it propagates.

Key features:
- **Starts at steady state**: All variables initialized to zero (representing no deviation from steady state)
- **One-time shock**: A single shock is applied at $t=4$ (first simulation period after initial conditions)
- **Shock magnitude**: Typically 1 standard deviation of the variable during the pandemic era (Q1 2020+)
- **Constants excluded**: The equation constants are excluded since we measure deviations from steady state

---

## Model Equations

### 1. Wage Equation

$$
gw_t = \sum_{i=1}^{4} \alpha_i \cdot gw_{t-i} + \sum_{i=1}^{4} \beta_i \cdot cf1_{t-i} + \sum_{i=1}^{4} \delta_i \cdot vu_{t-i} + \sum_{i=1}^{4} \phi_i \cdot diffcpicf_{t-i} + \sum_{i=1}^{4} \psi_i \cdot cu_{t-i}
$$

**Note**: Productivity ($magpty$) and COVID dummies are set to zero in the IRF.

### 2. Excess Demand (Endogenous)

$$
ed_t^{raw} = \ln(W_t) - \ln(NGDPPOT_t) - \ln(TCU^*/100)
$$

$$
ed_t = ed_t^{raw} - \bar{ed}_t^{40q}
$$

where the wage level evolves as:
$$
\ln(W_t) = \ln(W_{t-1}) + \frac{gw_t}{400}
$$

and nominal potential GDP evolves at a constant rate:
$$
\ln(NGDPPOT_t) = \ln(NGDPPOT_{t-1}) + \frac{g_{NGDPPOT}}{400}
$$

In the IRF (starting from steady state with $gw_t = 0$), both $W_t$ and $NGDPPOT_t$ remain at their initial levels, so excess demand deviations arise only from wage growth deviations.

### 3. Shortage Equation (Endogenous)

$$
shortage_t = \sum_{i=1}^{4} \phi_i \cdot shortage_{t-i} + \sum_{i=0}^{4} \psi_i \cdot ed_{t-i} + \sum_{i=0}^{4} \xi_i \cdot gscpi_{t-i}
$$

**Note**: The constant is excluded for IRF purposes. Shortages depend on the **endogenous** excess demand computed from simulated wages.

### 4. Price Equation

$$
gcpi_t = \sum_{i=1}^{4} \gamma_i \cdot gcpi_{t-i} + \sum_{i=0}^{4} \delta_i \cdot gw_{t-i} + \sum_{i=0}^{4} \phi_i \cdot grpe_{t-i} + \sum_{i=0}^{4} \psi_i \cdot grpf_{t-i} + \sum_{i=0}^{4} \xi_i \cdot shortage_{t-i}
$$

### 5. Catch-Up Term (Identity)

$$
diffcpicf_t = \frac{1}{4}(gcpi_t + gcpi_{t-1} + gcpi_{t-2} + gcpi_{t-3}) - cf1_{t-4}
$$

### 6. Expectations Equations

**Long-run (10-year):**
$$
cf10_t = \sum_{i=1}^{4} \alpha_i \cdot cf10_{t-i} + \sum_{i=0}^{4} \beta_i \cdot gcpi_{t-i}
$$

**Short-run (1-year):**
$$
cf1_t = \sum_{i=1}^{4} \alpha_i \cdot cf1_{t-i} + \sum_{i=0}^{4} \beta_i \cdot cf10_{t-i} + \sum_{i=0}^{4} \gamma_i \cdot gcpi_{t-i}
$$

---

## Shock Persistence

Shocks can be one-time (transitory) or persistent. The persistence is controlled by the autoregressive parameter $\rho$:

$$
x_t^{shock} = \rho \cdot x_{t-1}^{shock} + \epsilon_t
$$

where $\epsilon_t$ is the initial shock (applied at $t=4$ only).

| $\rho$ Value | Behavior |
|--------------|----------|
| $\rho = 0$ | One-time shock, no persistence |
| $\rho = 0.9$ | Highly persistent shock |
| $\rho = 1.0$ | Permanent shock (random walk) |

---

## Shocks Analyzed

### Original Bernanke-Blanchard Shocks

| Shock | Variable | Persistence | Transmission Channel |
|-------|----------|-------------|---------------------|
| Energy prices | $grpe$ | $\rho = 0$ | $grpe \to gcpi$ |
| Food prices | $grpf$ | $\rho = 0$ | $grpf \to gcpi$ |
| V/U ratio | $vu$ | $\rho = 1.0$ | $vu \to gw \to W \to ed \to shortage \to gcpi$ |
| Shortage | $shortage$ | $\rho = 0$ | $shortage \to gcpi$ |

### New Model Shocks

| Shock | Variable | Persistence | Transmission Channel |
|-------|----------|-------------|---------------------|
| Supply chain (GSCPI) | $gscpi$ | $\rho = 0$ or $0.9$ | $gscpi \to shortage \to gcpi$ |
| Capacity utilization | $cu$ | $\rho = 0$ or $0.9$ | $cu \to gw \to W \to ed \to shortage \to gcpi$ |

---

## Shock Magnitudes

Shock magnitudes are set to 1 standard deviation of the variable during the pandemic era (Q1 2020 onwards):

$$
shock\_val_x = \sigma_x^{Q1\,2020+}
$$

This ensures comparability across shocks and reflects the typical magnitude of disturbances observed during the pandemic.

---

## Simulation Algorithm

The key insight is that the simulation must be ordered correctly to capture the endogenous feedback: **wages first**, then **excess demand**, then **shortages**, then **prices**.

### Initialization

1. Set time horizon: $T = 32$ quarters (8 years)
2. Initialize all arrays to zero (steady state)
3. Initialize wage level: $\ln(W_0) = 0$ (normalized)
4. Initialize potential GDP level: $\ln(NGDPPOT_0) = 0$ (normalized)
5. First 4 periods ($t = 0, 1, 2, 3$) serve as initial conditions

### Simulation Loop

For $t = 4, 5, \ldots, T-1$:

**Step 1: Apply Shock**
$$
\epsilon_t = \begin{cases}
shock\_val & \text{if } t = 4 \\
0 & \text{otherwise}
\end{cases}
$$

**Step 2: Update Exogenous Shock Series with Persistence**
$$
x_t^{shock} = \rho \cdot x_{t-1}^{shock} + \epsilon_t
$$

for $x \in \{grpe, grpf, vu, gscpi, cu\}$.

**Step 3: Compute Wages** (computed first since excess demand depends on wages)
$$
gw_t = f(gw_{t-1:t-4}, cf1_{t-1:t-4}, vu_{t-1:t-4}^{shock}, diffcpicf_{t-1:t-4}, cu_{t-1:t-4}^{shock})
$$

**Step 4: Update Wage Level and Compute Excess Demand** (endogenous)
$$
\ln(W_t) = \ln(W_{t-1}) + \frac{gw_t}{400}
$$
$$
\ln(NGDPPOT_t) = \ln(NGDPPOT_{t-1}) + \frac{g_{NGDPPOT}}{400}
$$
$$
ed_t^{raw} = \ln(W_t) - \ln(NGDPPOT_t) - \ln(TCU^*/100)
$$

Compute rolling trend:
$$
\bar{ed}_t^{40q} = \frac{1}{\min(40, t+1)} \sum_{i=0}^{\min(39,t)} ed_{t-i}^{raw}
$$

Detrend:
$$
ed_t = ed_t^{raw} - \bar{ed}_t^{40q}
$$

**Step 5: Compute Shortage** (uses endogenous excess demand)
$$
shortage_t = f(shortage_{t-1:t-4}, ed_{t:t-4}, gscpi_{t:t-4}^{shock})
$$

*Exception*: If a direct shortage shock is applied, $shortage_t = shock\_val_{shortage}$ at $t=4$.

**Step 6: Compute Prices**
$$
gcpi_t = f(gcpi_{t-1:t-4}, gw_{t:t-4}, grpe_{t:t-4}^{shock}, grpf_{t:t-4}^{shock}, shortage_{t:t-4})
$$

**Step 7: Update Catch-up Term**
$$
diffcpicf_t = \frac{1}{4}\sum_{i=0}^{3} gcpi_{t-i} - cf1_{t-4}
$$

**Step 8: Update Expectations**
$$
cf10_t = f(cf10_{t-1:t-4}, gcpi_{t:t-4})
$$
$$
cf1_t = f(cf1_{t-1:t-4}, cf10_{t:t-4}, gcpi_{t:t-4})
$$

---

## Transmission Channels

### Direct Price Shocks (Energy, Food)

$$
grpe_t \text{ or } grpf_t \xrightarrow{\text{price eq.}} gcpi_t \xrightarrow{\text{expectations}} cf1_t, cf10_t
$$

These shocks feed directly into the price equation with no prior channels.

### Labor Market Shock (V/U)

The V/U shock now has an **extended transmission channel** through endogenous excess demand:

$$
vu_t \xrightarrow{\text{wage eq.}} gw_t \xrightarrow{\text{level}} W_t \xrightarrow{\text{ED}} ed_t \xrightarrow{\text{shortage eq.}} shortage_t \xrightarrow{\text{price eq.}} gcpi_t
$$

Additionally, V/U has the standard direct wage-to-price channel:
$$
vu_t \xrightarrow{\text{wage eq.}} gw_t \xrightarrow{\text{price eq.}} gcpi_t
$$

And the feedback through catch-up:
$$
gcpi_t \xrightarrow{\text{catch-up}} diffcpicf_t \xrightarrow{\text{wage eq.}} gw_{t+1}
$$

The V/U shock is persistent ($\rho = 1$), reflecting that labor market tightness is slow-moving.

### Shortage Shock

$$
shortage_t \xrightarrow{\text{price eq.}} gcpi_t \xrightarrow{\text{expectations}} cf1_t, cf10_t
$$

A direct shortage shock bypasses the shortage equation and affects prices immediately.

### Supply Chain Shock (GSCPI) [NEW]

$$
gscpi_t \xrightarrow{\text{shortage eq.}} shortage_t \xrightarrow{\text{price eq.}} gcpi_t
$$

Global supply chain pressure increases shortages, which affect prices.

### Capacity Utilization Shock [NEW]

Capacity utilization has **two transmission channels**:

**Channel 1 (Direct):** Through wages to prices
$$
cu_t \xrightarrow{\text{wage eq.}} gw_t \xrightarrow{\text{price eq.}} gcpi_t
$$

**Channel 2 (Indirect):** Through wages to excess demand to shortages
$$
cu_t \xrightarrow{\text{wage eq.}} gw_t \xrightarrow{\text{level}} W_t \xrightarrow{\text{ED}} ed_t \xrightarrow{\text{shortage eq.}} shortage_t \xrightarrow{\text{price eq.}} gcpi_t
$$

Higher capacity utilization puts upward pressure on wages, which increases wage levels, which increases excess demand, which increases shortages, which finally raises prices.

---

## The Key Feedback Loop

The critical innovation in the new model is the **endogenous excess demand feedback loop**:

$$
\boxed{gw_t \to W_t \to ed_t \to shortage_t \to gcpi_t}
$$

This means that any shock affecting wages (V/U, capacity utilization, catch-up effects) will also affect excess demand and hence shortages. The full transmission path is:

1. **Wage shock** increases wage growth ($gw_t$)
2. **Wage level** accumulates: $W_t$ rises
3. **Excess demand** increases: $ed_t = \ln(W_t) - \ln(NGDPPOT_t) - \ln(TCU^*/100) - trend$
4. **Shortages** respond to excess demand via the shortage equation
5. **Inflation** responds to both wages (direct) and shortages (indirect)

This captures the intuition that strong wage growth creates demand pressure that exceeds supply capacity, leading to shortages and higher prices.

---

## Comparison with Decomposition and Conditional Forecast

| Aspect | IRF | Decomposition | Conditional Forecast |
|--------|-----|---------------|---------------------|
| **Purpose** | Shock propagation | Historical attribution | Forward projection |
| **Initial conditions** | Steady state (zeros) | Historical data | Historical data |
| **Shock treatment** | One shock at a time | All shocks active | Scenario-based |
| **Constants** | Excluded | Included | Adjusted for convergence |
| **Excess demand** | Endogenous (from wages) | Endogenous (from wages) | Endogenous (from wages) |
| **Time horizon** | 32 quarters | Historical sample | 100+ years |

All three simulations now compute excess demand **endogenously** from simulated wage levels, ensuring the full feedback loop is captured consistently across the analysis.

---

## Output Files

The simulation generates the following Excel files:

### Original BB Shocks
- `results_energy.xlsx`: Energy price shock ($\rho = 0$)
- `results_food.xlsx`: Food price shock ($\rho = 0$)
- `results_vu.xlsx`: V/U shock ($\rho = 1.0$, permanent)
- `results_shortage.xlsx`: Direct shortage shock ($\rho = 0$)

### New Model Shocks (One-time)
- `results_gscpi.xlsx`: GSCPI shock ($\rho = 0$)
- `results_gcu.xlsx`: Capacity utilization shock ($\rho = 0$)

### New Model Shocks (Persistent)
- `results_gscpi_persistent.xlsx`: GSCPI shock ($\rho = 0.9$)
- `results_gcu_persistent.xlsx`: Capacity utilization shock ($\rho = 0.9$)

Each file contains:
- `period`: Time index (1 to 32)
- `gw_simul`: Wage growth response
- `gcpi_simul`: CPI inflation response
- `shortage_simul`: Shortage response
- `ed_simul`: Excess demand response (endogenous)
- `cf1_simul`: 1-year expectations response
- `cf10_simul`: 10-year expectations response
- `diffcpicf_simul`: Catch-up term response
- `log_w_simul`: Log wage level
- Shock series for each exogenous variable

---

## Interpreting Results

### Peak Response

The peak inflation response indicates the maximum impact of the shock:
- **Timing**: Earlier peaks indicate faster transmission
- **Magnitude**: Larger peaks indicate stronger effects

### Persistence

The rate at which responses decay back to zero reflects:
- The shock's own persistence ($\rho$)
- The model's internal dynamics (autoregressive coefficients)
- Feedback effects (expectations, catch-up term, excess demand)

### Comparison Across Shocks

Comparing peak responses across shocks (all normalized to 1 standard deviation) reveals which shocks have the largest inflationary impact.

### Wage-to-Shortage Channel

With endogenous excess demand, shocks to V/U and capacity utilization now have an additional inflationary channel through shortages. Compare the shortage response across different shocks to understand the relative importance of this channel.

---

## References

- Bernanke, B. S., & Blanchard, O. J. (2023). What Caused the U.S. Pandemic-Era Inflation? *Brookings Papers on Economic Activity*.
- Liang & Sun (2025). Extended model with endogenous shortages.
