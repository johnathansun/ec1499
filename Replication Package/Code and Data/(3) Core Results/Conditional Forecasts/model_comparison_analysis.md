# Model Comparison: Original Bernanke-Blanchard vs New Model

## Overview

This document analyzes the key differences between the original Bernanke-Blanchard (2023) conditional forecasts and our modified model with endogenous shortages. Both models use the v/u → 1.2 scenario.

---

## Observed Differences

### Inflation (gcpi)

| Period | BB Model | New Model | Difference |
|--------|----------|-----------|------------|
| 2023 Q2 | 3.46% | 4.43% | +0.97 pp |
| 2023 Q3 | 3.43% | 4.50% | +1.08 pp |
| 2023 Q4 | 3.13% | 4.10% | +0.97 pp |
| 2024 Q1 | 2.88% | 2.64% | -0.24 pp |
| 2025 Q1 | 2.84% | 2.73% | -0.11 pp |
| 2027 Q1 | 2.50% | 2.39% | -0.11 pp |
| **Terminal** | **2.39%** | **1.37%** | **-1.02 pp** |

**Pattern:** New model predicts ~1 pp higher inflation in 2023, but ~1 pp lower terminal inflation.

### Wage Growth (gw)

| Period | BB Model | New Model | Difference |
|--------|----------|-----------|------------|
| 2023 Q3 | 3.54% | 3.27% | -0.27 pp |
| 2025 Q1 | 2.67% | 2.55% | -0.11 pp |
| **Terminal** | **2.48%** | **1.57%** | **-0.91 pp** |

**Pattern:** New model predicts consistently lower wage growth, especially in the long run.

### Shortage

| Period | BB Model (exogenous) | New Model (endogenous) | Excess Demand |
|--------|----------------------|------------------------|---------------|
| 2023 Q2 | 10.00 | 19.12 | +0.001 |
| 2023 Q3 | 10.00 | 23.40 | -0.000 |
| 2024 Q1 | 10.00 | 10.24 | -0.003 |
| 2025 Q1 | 10.00 | 12.13 | -0.012 |
| 2027 Q1 | 10.00 | 10.60 | -0.029 |
| **Terminal** | **10.00** | **5.87** | **-0.055** |

**Pattern:**
- BB holds shortage fixed at 10.0 throughout
- New model: shortage elevated in 2023 (~19-23), then falls below 10 as excess demand becomes negative

---

## Mechanisms Explaining the Differences

### 1. Higher Inflation in 2023 (New Model)

**Root cause:** Endogenous shortage amplification

In the new model, shortage is determined by the equation:

$$
shortage_t = \alpha_0 + \sum_{i=1}^{4} \phi_i \cdot shortage_{t-i} + \sum_{i=0}^{4} \psi_i \cdot ed_{t-i} + \sum_{i=0}^{4} \xi_i \cdot gscpi_{t-i}
$$

**Key coefficient:** Sum of excess demand coefficients = **+15.4**

**Transmission mechanism:**
```
Initial conditions: excess_demand ≈ 0 (near trend)
         ↓
Positive ED coefficient (+15.4) keeps shortage elevated
         ↓
Shortage stays at ~19-23 (vs BB's immediate drop to 10)
         ↓
Higher shortage × price equation coefficient (+0.019)
         ↓
Higher inflation in 2023 (+1 pp vs BB)
```

In BB, shortage **immediately drops to the steady-state value of 10** at the start of the forecast. In our model, the endogenous shortage equation allows shortage to **remain elevated** based on the initial excess demand conditions.

### 2. Lower Terminal Inflation (New Model)

**Root cause:** Excess demand becomes increasingly negative over time

**The wage-capacity gap mechanism:**

In the conditional forecast, we assume:
- Nominal potential GDP growth: $g_{NGDPPOT} = 4\%$ (annualized)
- Terminal wage growth: $g_w \approx 2.5\%$

Since wages grow **slower** than nominal potential GDP:

$$
ed_t = \ln(W_t) - \ln(NGDPPOT_t) - \ln(TCU_t/100) - \overline{ed}_t
$$

$$
\frac{d(ed)}{dt} \approx \frac{g_w - g_{NGDPPOT}}{400} = \frac{2.5 - 4.0}{400} < 0
$$

**Transmission mechanism:**
```
Wage growth (2.5%) < Potential GDP growth (4%)
         ↓
Excess demand becomes increasingly negative (-0.05 by 2030)
         ↓
Negative ED → shortage falls via shortage equation
         ↓
Terminal shortage = 5.87 (vs BB's fixed 10)
         ↓
Lower shortage → lower inflation (1.37% vs 2.39%)
```

This is a **disinflationary force unique to our model** - the wage-capacity gap creates sustained downward pressure on shortages and inflation.

### 3. Lower Terminal Wage Growth (New Model)

**Two channels:**

1. **Lower inflation expectations:**
   - Lower terminal inflation → lower expected inflation
   - Lower expected inflation → lower wage demands (via Phillips curve)

2. **Dampened wage-price spiral:**
   - The shortage channel creates additional feedback
   - Lower wages → lower excess demand → lower shortage → lower inflation → lower wage expectations
   - This dampens the traditional wage-price spiral

---

## Key Model Coefficients

### Shortage Equation (New Model Only)

| Variable | Coefficient | Interpretation |
|----------|-------------|----------------|
| const | 4.16 | Baseline shortage level |
| Sum(lagged shortage) | 0.59 | Persistence (mean-reversion) |
| Sum(excess_demand) | **+15.38** | Positive ED strongly increases shortage |
| Sum(gscpi) | 3.45 | Supply chain pressure raises shortage |

**Implied steady-state shortage** (when ED = 0, GSCPI = 0):

$$
shortage^* = \frac{4.16}{1 - 0.59} = 10.24
$$

This is similar to BB's assumed shortage* = 10, but **falls when ED becomes negative**.

### Price Equation

| Variable | Sum of Coefficients |
|----------|---------------------|
| Shortage (L0-L4) | +0.019 |
| Wages (L0-L4) | ~0.65 |

### Wage Equation

| Variable | Sum of Coefficients |
|----------|---------------------|
| Capacity Utilization (L1-L4) | -4.20 |
| V/U ratio (L1-L4) | ~0.25 |

---

## Summary

| Feature | BB Model | New Model | Economic Interpretation |
|---------|----------|-----------|------------------------|
| **Shortage treatment** | Exogenous (fixed at 10) | Endogenous (depends on ED, GSCPI) | Shortages respond to demand conditions |
| **Short-run inflation** | Lower | Higher (+1 pp in 2023) | Shortage stays elevated longer |
| **Long-run inflation** | Higher (2.39%) | Lower (1.37%) | Wage-capacity gap is disinflationary |
| **Long-run wages** | Higher (2.48%) | Lower (1.57%) | Dampened wage-price spiral |
| **Terminal shortage** | 10.0 | 5.87 | ED < 0 pushes shortage down |

### Bottom Line

The **endogenous shortage mechanism** creates asymmetric dynamics:

- **Short-run:** More inflationary (shortage amplifies initial conditions)
- **Long-run:** More disinflationary (wage-capacity gap reduces shortage below steady state)

This explains why our model forecasts higher inflation in 2023 but lower terminal inflation than the original Bernanke-Blanchard model.

---

## Data Sources

- BB Model: `Output Data Python/terminal_mid_og.xlsx`
- New Model: `Output Data Python (New Model)/Output Data (New Detrended ED)/terminal_mid.xlsx`
- Coefficients: `(2) Regressions/Output Data (New Detrended ED)/eq_coefficients_new_model.xlsx`
