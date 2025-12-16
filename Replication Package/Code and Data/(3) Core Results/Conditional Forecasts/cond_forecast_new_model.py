# %%
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modified Bernanke-Blanchard Model - Conditional Forecasts
Liang & Sun (2025)

This script runs conditional forecasts using the modified model with:
1. Capacity utilization in the wage equation
2. Endogenous shortage equation (depends on excess demand and GSCPI)

We run conditional forecasts using three assumptions for v/u:
1. v/u declines to 0.8 after 8 quarters (tight labor market normalization)
2. v/u declines to 1.2 after 8 quarters (target steady state)
3. v/u remains at 1.8 (persistent tight labor market)

Key difference from BB: shortage is now endogenous, determined by:
- Lagged shortage (persistence)
- Excess demand (log wages - log potential GDP - log capacity utilization)
- GSCPI (supply chain pressure)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from dateutil.relativedelta import relativedelta

# %%

#****************************CHANGE PATH HERE************************************
# Input Location - coefficients and data from regression_new_model.py
coef_path = Path("/Users/johnathansun/Documents/ec1499/Replication Package/Code and Data/(2) Regressions/Output Data Python (New Model)/eq_coefficients_new_model.xlsx")
data_path = Path("/Users/johnathansun/Documents/ec1499/Replication Package/Code and Data/(2) Regressions/Output Data Python (New Model)/eq_simulations_data_new_model.xlsx")

# Output Location
output_dir = Path("/Users/johnathansun/Documents/ec1499/Replication Package/Code and Data/(3) Core Results/Conditional Forecasts/Output Data Python (New Model)")
output_dir.mkdir(parents=True, exist_ok=True)

#*********************************************************************************

print("="*80)
print("CONDITIONAL FORECASTS - MODIFIED MODEL")
print("Liang & Sun (2025)")
print("="*80)

print("\nLoading data and coefficients...")

# Load simulation data
data_orig = pd.read_excel(data_path)

# Convert period to datetime if needed
if not pd.api.types.is_datetime64_any_dtype(data_orig['period']):
    data_orig['period'] = pd.to_datetime(data_orig['period'])

# Filter data from Q2 2022 onwards for conditional forecast
data = data_orig[data_orig['period'] >= '2022-04-01'].copy().reset_index(drop=True)

print(f"Data loaded: {len(data)} observations starting from {data['period'].min()}")


# %%
def load_coefficients(coef_path):
    """Load coefficients from Excel file for new model"""
    gw_beta = pd.read_excel(coef_path, sheet_name='gw')
    shortage_beta = pd.read_excel(coef_path, sheet_name='shortage')
    gcpi_beta = pd.read_excel(coef_path, sheet_name='gcpi')
    cf1_beta = pd.read_excel(coef_path, sheet_name='cf1')
    cf10_beta = pd.read_excel(coef_path, sheet_name='cf10')

    return {
        'gw': gw_beta['beta'].values,
        'shortage': shortage_beta['beta'].values,
        'gcpi': gcpi_beta['beta'].values,
        'cf1': cf1_beta['beta'].values,
        'cf10': cf10_beta['beta'].values
    }


def calculate_wage_constant_new(gw_beta, gcpi_beta, magpty_star, shortage_star, vu_star, cu_star):
    """
    Calculate the wage constant adjustment for steady state targeting.
    Modified for new model with capacity utilization.

    In the new model, the wage equation includes gcu terms (indices 18-21 after constant).
    """
    # Sum of vu coefficients (indices 10-13 in gw equation, after constant)
    sum_vu_coef = sum(gw_beta[10:14])

    # Sum of cu coefficients (indices 18-21 in gw equation, after constant)
    sum_cu_coef = sum(gw_beta[18:22])

    # Sum of gw lag coefficients (indices 1-4)
    sum_gw_lag_coef = sum(gw_beta[1:5])

    # Sum of gcpi lag coefficients (indices 2-5 in gcpi equation)
    sum_gcpi_lag_coef = sum(gcpi_beta[2:6])

    # Sum of shortage coefficients (indices 21-25 in gcpi equation)
    sum_shortage_coef = sum(gcpi_beta[21:26])

    # magpty coefficient in gcpi equation (index 1)
    magpty_coef_gcpi = gcpi_beta[1]

    # magpty coefficient in gw equation (index 9)
    magpty_coef_gw = gw_beta[9]

    # gcpi constant (index 0)
    gcpi_const = gcpi_beta[0]

    wage_constant = (
        (-sum_vu_coef * vu_star) -
        (sum_cu_coef * cu_star) -  # NEW: capacity utilization term
        (1 - sum_gw_lag_coef) * (
            (magpty_coef_gcpi / (1 - sum_gcpi_lag_coef) +
             magpty_coef_gw / (1 - sum_gw_lag_coef)) * magpty_star +
            sum_shortage_coef / (1 - sum_gcpi_lag_coef) * shortage_star +
            gcpi_const / (1 - sum_gcpi_lag_coef)
        )
    )

    return wage_constant


def conditional_forecast_new_model(data, coef_path, wage_constant, magpty_star, gscpi_star,
                                    vu_decline_val, vu_quarters_decline, cu_star=0,
                                    excess_demand_star=None, years=100):
    """
    Run conditional forecast with the new model (endogenous shortage).

    Parameters:
    -----------
    data : DataFrame
        Historical data starting from forecast origin
    coef_path : Path
        Path to coefficients Excel file
    wage_constant : float
        Adjusted wage constant for steady state
    magpty_star : float
        Long-run productivity growth target
    gscpi_star : float
        Long-run GSCPI target (steady state = 0, no supply chain pressure)
    vu_decline_val : float
        Terminal value of v/u ratio
    vu_quarters_decline : int
        Number of quarters for v/u to reach terminal value
    cu_star : float
        Long-run detrended capacity utilization (steady state = 0, i.e. on trend)
    excess_demand_star : float
        Long-run excess demand (if None, use last historical value)
    years : int
        Forecast horizon in years

    Returns:
    --------
    out_data : DataFrame
        Forecast results including endogenous shortage
    """

    # Load coefficients
    coeffs = load_coefficients(coef_path)
    gw_beta = coeffs['gw']
    shortage_beta = coeffs['shortage']
    gcpi_beta = coeffs['gcpi']
    cf1_beta = coeffs['cf1']
    cf10_beta = coeffs['cf10']

    print(f"  Running forecast: v/u -> {vu_decline_val} over {vu_quarters_decline} quarters")

    # Define forecast horizon
    timesteps = len(data) + 4 * years

    # Initialize period array
    period = list(data['period'].values)
    last_period = pd.Timestamp(period[-1])
    for i in range(4 * years):
        last_period = last_period + relativedelta(months=3)
        period.append(last_period)
    period = np.array(period)

    # Extract historical data
    gw = data['gw'].values
    gcpi = data['gcpi'].values
    cf1 = data['cf1'].values
    cf10 = data['cf10'].values
    diffcpicf = data['diffcpicf'].values

    grpe = data['grpe'].values
    grpf = data['grpf'].values
    vu = data['vu'].values
    shortage = data['shortage'].values
    magpty = data['magpty'].values

    # New model variables
    cu = data['cu'].values
    gscpi = data['gscpi'].values
    excess_demand = data['excess_demand'].values

    # Initialize levels for endogenous excess demand calculation
    # excess_demand = log(W) - log(NGDPPOT) - log(TCU/100) - trend
    log_w_init = data['log_w'].values[-1] if 'log_w' in data.columns else np.log(150)  # ECI index ~150
    log_ngdppot_init = data['log_ngdppot'].values[-1] if 'log_ngdppot' in data.columns else np.log(25000)
    log_tcu_star = np.log(0.75)  # Steady state TCU = 75%
    g_ngdppot = 4.0  # Nominal potential GDP growth rate (annualized)

    # Get the trend value (last historical value of the rolling mean)
    # In steady state, we assume trend is fixed
    if 'excess_demand_trend' in data.columns:
        excess_demand_trend = data['excess_demand_trend'].values[-1]
    else:
        # Approximate: raw excess demand at last period minus detrended value
        raw_ed_last = log_w_init - log_ngdppot_init - np.log(data['tcu'].values[-1]/100) if 'tcu' in data.columns else 0
        excess_demand_trend = raw_ed_last - excess_demand[-1] if len(excess_demand) > 0 else 0

    # Initialize simulation arrays (first 4 values from historical data)
    gw_simul = np.zeros(timesteps)
    gcpi_simul = np.zeros(timesteps)
    cf1_simul = np.zeros(timesteps)
    cf10_simul = np.zeros(timesteps)
    diffcpicf_simul = np.zeros(timesteps)
    shortage_simul = np.zeros(timesteps)

    gw_simul[:4] = gw[:4]
    gcpi_simul[:4] = gcpi[:4]
    cf1_simul[:4] = cf1[:4]
    cf10_simul[:4] = cf10[:4]
    diffcpicf_simul[:4] = diffcpicf[:4]
    shortage_simul[:4] = shortage[:4]

    # Initialize exogenous variables
    grpe_simul = np.zeros(timesteps)
    grpf_simul = np.zeros(timesteps)
    vu_simul = np.zeros(timesteps)
    magpty_simul = np.zeros(timesteps)
    cu_simul = np.zeros(timesteps)
    gscpi_simul = np.zeros(timesteps)
    excess_demand_simul = np.zeros(timesteps)

    # Track levels for endogenous excess demand
    log_w_simul = np.zeros(timesteps)
    log_ngdppot_simul = np.zeros(timesteps)
    raw_excess_demand_simul = np.zeros(timesteps)  # For computing rolling trend

    grpe_simul[:len(grpe)] = grpe
    grpf_simul[:len(grpf)] = grpf
    vu_simul[:len(vu)] = vu
    magpty_simul[:len(magpty)] = magpty
    cu_simul[:len(cu)] = np.nan_to_num(cu, nan=0.0)
    gscpi_simul[:len(gscpi)] = np.nan_to_num(gscpi, nan=0.0)
    excess_demand_simul[:len(excess_demand)] = np.nan_to_num(excess_demand, nan=0.0)

    # Initialize level arrays from historical data
    # Reconstruct log_w from cumulating wage growth backwards from last known value
    log_w_simul[3] = log_w_init
    for i in range(2, -1, -1):
        log_w_simul[i] = log_w_simul[i+1] - gw[i+1] / 400

    log_ngdppot_simul[3] = log_ngdppot_init
    for i in range(2, -1, -1):
        log_ngdppot_simul[i] = log_ngdppot_simul[i+1] - g_ngdppot / 400

    # Initialize raw excess demand for first 4 periods
    for i in range(4):
        raw_excess_demand_simul[i] = log_w_simul[i] - log_ngdppot_simul[i] - log_tcu_star

    # Calculate slope for v/u transition
    vu_slope = (vu_decline_val - vu_simul[3]) / vu_quarters_decline

    # Run forecast
    for t in range(4, timesteps):

        # Set exogenous variables to steady state values
        grpe_simul[t] = 0
        grpf_simul[t] = 0
        magpty_simul[t] = magpty_star
        cu_simul[t] = cu_star
        gscpi_simul[t] = gscpi_star

        # V/U path: gradual transition to terminal value
        if vu_slope >= 0:
            vu_simul[t] = min(vu_decline_val, vu_simul[t-1] + vu_slope)
        else:
            vu_simul[t] = max(vu_decline_val, vu_simul[t-1] + vu_slope)

        # STEP 1: WAGE EQUATION (compute first since excess demand depends on wages)
        # Indices: const=0, L1-L4 gw=1-4, L1-L4 cf1=5-8, magpty=9,
        #          L1-L4 vu=10-13, L1-L4 diffcpicf=14-17, L1-L4 cu=18-21, dummies=22-23
        gw_simul[t] = (
            gw_beta[1] * gw_simul[t-1] +
            gw_beta[2] * gw_simul[t-2] +
            gw_beta[3] * gw_simul[t-3] +
            gw_beta[4] * gw_simul[t-4] +
            gw_beta[5] * cf1_simul[t-1] +
            gw_beta[6] * cf1_simul[t-2] +
            gw_beta[7] * cf1_simul[t-3] +
            gw_beta[8] * cf1_simul[t-4] +
            gw_beta[9] * magpty_simul[t-1] +
            gw_beta[10] * vu_simul[t-1] +
            gw_beta[11] * vu_simul[t-2] +
            gw_beta[12] * vu_simul[t-3] +
            gw_beta[13] * vu_simul[t-4] +
            gw_beta[14] * diffcpicf_simul[t-1] +
            gw_beta[15] * diffcpicf_simul[t-2] +
            gw_beta[16] * diffcpicf_simul[t-3] +
            gw_beta[17] * diffcpicf_simul[t-4] +
            gw_beta[18] * cu_simul[t-1] +
            gw_beta[19] * cu_simul[t-2] +
            gw_beta[20] * cu_simul[t-3] +
            gw_beta[21] * cu_simul[t-4] +
            wage_constant
        )

        # STEP 2: UPDATE LEVELS AND COMPUTE EXCESS DEMAND (endogenous)
        # Update wage level: log(W_t) = log(W_{t-1}) + gw_t / 400
        log_w_simul[t] = log_w_simul[t-1] + gw_simul[t] / 400

        # Update potential GDP level (exogenous growth)
        log_ngdppot_simul[t] = log_ngdppot_simul[t-1] + g_ngdppot / 400

        # Compute raw excess demand from levels
        raw_excess_demand_simul[t] = log_w_simul[t] - log_ngdppot_simul[t] - log_tcu_star

        # Compute trend as 40-quarter rolling mean (same as regression)
        # Use available history, minimum 4 quarters
        lookback = min(40, t + 1)
        rolling_trend = np.mean(raw_excess_demand_simul[t-lookback+1:t+1])

        # Detrend excess demand
        excess_demand_simul[t] = raw_excess_demand_simul[t] - rolling_trend

        # STEP 3: SHORTAGE EQUATION (uses endogenous excess demand)
        # Indices: const=0, L1-L4 shortage=1-4, excess_demand=5-9, gscpi=10-14
        shortage_simul[t] = (
            shortage_beta[0] +  # constant
            shortage_beta[1] * shortage_simul[t-1] +
            shortage_beta[2] * shortage_simul[t-2] +
            shortage_beta[3] * shortage_simul[t-3] +
            shortage_beta[4] * shortage_simul[t-4] +
            shortage_beta[5] * excess_demand_simul[t] +
            shortage_beta[6] * excess_demand_simul[t-1] +
            shortage_beta[7] * excess_demand_simul[t-2] +
            shortage_beta[8] * excess_demand_simul[t-3] +
            shortage_beta[9] * excess_demand_simul[t-4] +
            shortage_beta[10] * gscpi_simul[t] +
            shortage_beta[11] * gscpi_simul[t-1] +
            shortage_beta[12] * gscpi_simul[t-2] +
            shortage_beta[13] * gscpi_simul[t-3] +
            shortage_beta[14] * gscpi_simul[t-4]
        )

        # STEP 4: PRICE EQUATION (uses endogenous shortage from step 3)
        gcpi_simul[t] = (
            gcpi_beta[1] * magpty_simul[t] +
            gcpi_beta[2] * gcpi_simul[t-1] +
            gcpi_beta[3] * gcpi_simul[t-2] +
            gcpi_beta[4] * gcpi_simul[t-3] +
            gcpi_beta[5] * gcpi_simul[t-4] +
            gcpi_beta[6] * gw_simul[t] +
            gcpi_beta[7] * gw_simul[t-1] +
            gcpi_beta[8] * gw_simul[t-2] +
            gcpi_beta[9] * gw_simul[t-3] +
            gcpi_beta[10] * gw_simul[t-4] +
            gcpi_beta[11] * grpe_simul[t] +
            gcpi_beta[12] * grpe_simul[t-1] +
            gcpi_beta[13] * grpe_simul[t-2] +
            gcpi_beta[14] * grpe_simul[t-3] +
            gcpi_beta[15] * grpe_simul[t-4] +
            gcpi_beta[16] * grpf_simul[t] +
            gcpi_beta[17] * grpf_simul[t-1] +
            gcpi_beta[18] * grpf_simul[t-2] +
            gcpi_beta[19] * grpf_simul[t-3] +
            gcpi_beta[20] * grpf_simul[t-4] +
            gcpi_beta[21] * shortage_simul[t] +  # Uses ENDOGENOUS shortage
            gcpi_beta[22] * shortage_simul[t-1] +
            gcpi_beta[23] * shortage_simul[t-2] +
            gcpi_beta[24] * shortage_simul[t-3] +
            gcpi_beta[25] * shortage_simul[t-4] +
            gcpi_beta[0]
        )

        # Catch-up term
        diffcpicf_simul[t] = 0.25 * (gcpi_simul[t] + gcpi_simul[t-1] +
                                     gcpi_simul[t-2] + gcpi_simul[t-3]) - cf1_simul[t-4]

        # Long-run expectations (cf10)
        cf10_simul[t] = (
            cf10_beta[0] * cf10_simul[t-1] +
            cf10_beta[1] * cf10_simul[t-2] +
            cf10_beta[2] * cf10_simul[t-3] +
            cf10_beta[3] * cf10_simul[t-4] +
            cf10_beta[4] * gcpi_simul[t] +
            cf10_beta[5] * gcpi_simul[t-1] +
            cf10_beta[6] * gcpi_simul[t-2] +
            cf10_beta[7] * gcpi_simul[t-3] +
            cf10_beta[8] * gcpi_simul[t-4]
        )

        # Short-run expectations (cf1)
        cf1_simul[t] = (
            cf1_beta[0] * cf1_simul[t-1] +
            cf1_beta[1] * cf1_simul[t-2] +
            cf1_beta[2] * cf1_simul[t-3] +
            cf1_beta[3] * cf1_simul[t-4] +
            cf1_beta[4] * cf10_simul[t] +
            cf1_beta[5] * cf10_simul[t-1] +
            cf1_beta[6] * cf10_simul[t-2] +
            cf1_beta[7] * cf10_simul[t-3] +
            cf1_beta[8] * cf10_simul[t-4] +
            cf1_beta[9] * gcpi_simul[t] +
            cf1_beta[10] * gcpi_simul[t-1] +
            cf1_beta[11] * gcpi_simul[t-2] +
            cf1_beta[12] * gcpi_simul[t-3] +
            cf1_beta[13] * gcpi_simul[t-4]
        )

    # Create quarter labels
    def period_to_quarter_label(dt):
        if pd.isna(dt):
            return ''
        dt = pd.Timestamp(dt)
        return f"{dt.year} Q{(dt.month - 1) // 3 + 1}"

    qtr_lbls = [period_to_quarter_label(p) for p in period]

    # Create output DataFrame
    out_data = pd.DataFrame({
        'period': period,
        'qtr_lbls': qtr_lbls,
        'gw_simul': gw_simul,
        'gcpi_simul': gcpi_simul,
        'cf1_simul': cf1_simul,
        'cf10_simul': cf10_simul,
        'grpe_simul': grpe_simul,
        'grpf_simul': grpf_simul,
        'vu_simul': vu_simul,
        'shortage_simul': shortage_simul,  # Now endogenous!
        'cu_simul': cu_simul,
        'gscpi_simul': gscpi_simul,
        'excess_demand_simul': excess_demand_simul
    })

    return out_data


# %%
print("\n" + "="*80)
print("CALCULATING WAGE CONSTANT (New Model)")
print("="*80)

# Load coefficients for wage constant calculation
coeffs = load_coefficients(coef_path)
gw_beta = coeffs['gw']
gcpi_beta = coeffs['gcpi']
shortage_beta = coeffs['shortage']

# Steady state parameters
magpty_star = 1.0      # Long-run productivity growth
gscpi_star = 0.0       # Long-run GSCPI (no supply chain pressure)
cu_star = 0.0          # Long-run detrended capacity utilization (on trend)
vu_star = 1.2          # Long-run target v/u

# Calculate long-run shortage implied by excess demand and gscpi at steady state
# In steady state: shortage_star = (sum_excess_demand * ed_star + sum_gscpi * gscpi_star) / (1 - sum_lag)
sum_shortage_lag = sum(shortage_beta[1:5])
sum_excess_demand = sum(shortage_beta[5:10])
sum_gscpi = sum(shortage_beta[10:15])

# Steady state excess demand = 0 (on trend)
excess_demand_star = 0.0

if abs(1 - sum_shortage_lag) > 0.01:
    shortage_star = (shortage_beta[0] + sum_excess_demand * excess_demand_star + sum_gscpi * gscpi_star) / (1 - sum_shortage_lag)
else:
    shortage_star = 10.0  # Default

print(f"Steady state parameters:")
print(f"  magpty*:        {magpty_star:.2f}")
print(f"  vu*:            {vu_star:.2f}")
print(f"  cu*:            {cu_star:.2f}")
print(f"  gscpi*:         {gscpi_star:.2f}")
print(f"  excess_demand*: {excess_demand_star:.4f}")
print(f"  shortage* (implied): {shortage_star:.2f}")

wage_constant = calculate_wage_constant_new(gw_beta, gcpi_beta, magpty_star, shortage_star, vu_star, cu_star)
print(f"\nWage constant adjustment: {wage_constant:.6f}")


# %%
print("\n" + "="*80)
print("RUNNING CONDITIONAL FORECASTS (New Model)")
print("="*80)

# Scenario 1: v/u declines to 0.8
print("\nScenario 1: v/u -> 0.8")
result_low = conditional_forecast_new_model(
    data, coef_path, wage_constant, magpty_star, gscpi_star,
    vu_decline_val=0.8,
    vu_quarters_decline=8,
    cu_star=cu_star,
    excess_demand_star=excess_demand_star,
    years=100
)

# Scenario 2: v/u declines to 1.2 (target)
print("\nScenario 2: v/u -> 1.2 (target)")
result_mid = conditional_forecast_new_model(
    data, coef_path, wage_constant, magpty_star, gscpi_star,
    vu_decline_val=1.2,
    vu_quarters_decline=8,
    cu_star=cu_star,
    excess_demand_star=excess_demand_star,
    years=100
)

# Scenario 3: v/u stays at 1.8
print("\nScenario 3: v/u -> 1.8")
result_high = conditional_forecast_new_model(
    data, coef_path, wage_constant, magpty_star, gscpi_star,
    vu_decline_val=1.8,
    vu_quarters_decline=8,
    cu_star=cu_star,
    excess_demand_star=excess_demand_star,
    years=100
)


# %%
print("\n" + "="*80)
print("EXPORTING RESULTS")
print("="*80)

# Export to Excel
result_low.to_excel(output_dir / 'terminal_low.xlsx', index=False)
result_mid.to_excel(output_dir / 'terminal_mid.xlsx', index=False)
result_high.to_excel(output_dir / 'terminal_high.xlsx', index=False)

print(f"\nResults saved to: {output_dir}")
print("  - terminal_low.xlsx (v/u -> 0.8)")
print("  - terminal_mid.xlsx (v/u -> 1.2)")
print("  - terminal_high.xlsx (v/u -> 1.8)")


# %%
# Print summary
print("\n" + "="*80)
print("FORECAST SUMMARY (New Model)")
print("="*80)

# Filter to near-term forecast (2023-2027)
mask_low = (result_low['period'] >= '2023-01-01') & (result_low['period'] <= '2027-01-01')
mask_mid = (result_mid['period'] >= '2023-01-01') & (result_mid['period'] <= '2027-01-01')
mask_high = (result_high['period'] >= '2023-01-01') & (result_high['period'] <= '2027-01-01')

print(f"\nInflation forecasts (gcpi) for 2023-2027:")
print(f"\n  v/u -> 0.8:")
print(f"    2023 Q4: {result_low.loc[mask_low, 'gcpi_simul'].iloc[3]:.2f}%")
print(f"    2025 Q4: {result_low.loc[mask_low, 'gcpi_simul'].iloc[11]:.2f}%")
print(f"    Terminal: {result_low['gcpi_simul'].iloc[-1]:.2f}%")

print(f"\n  v/u -> 1.2:")
print(f"    2023 Q4: {result_mid.loc[mask_mid, 'gcpi_simul'].iloc[3]:.2f}%")
print(f"    2025 Q4: {result_mid.loc[mask_mid, 'gcpi_simul'].iloc[11]:.2f}%")
print(f"    Terminal: {result_mid['gcpi_simul'].iloc[-1]:.2f}%")

print(f"\n  v/u -> 1.8:")
print(f"    2023 Q4: {result_high.loc[mask_high, 'gcpi_simul'].iloc[3]:.2f}%")
print(f"    2025 Q4: {result_high.loc[mask_high, 'gcpi_simul'].iloc[11]:.2f}%")
print(f"    Terminal: {result_high['gcpi_simul'].iloc[-1]:.2f}%")

print(f"\nShortage forecasts (endogenous) for 2023-2027:")
print(f"\n  v/u -> 0.8:")
print(f"    2023 Q4: {result_low.loc[mask_low, 'shortage_simul'].iloc[3]:.2f}")
print(f"    Terminal: {result_low['shortage_simul'].iloc[-1]:.2f}")

print(f"\n  v/u -> 1.2:")
print(f"    2023 Q4: {result_mid.loc[mask_mid, 'shortage_simul'].iloc[3]:.2f}")
print(f"    Terminal: {result_mid['shortage_simul'].iloc[-1]:.2f}")

print(f"\n  v/u -> 1.8:")
print(f"    2023 Q4: {result_high.loc[mask_high, 'shortage_simul'].iloc[3]:.2f}")
print(f"    Terminal: {result_high['shortage_simul'].iloc[-1]:.2f}")

print("\n" + "="*80)
print("CONDITIONAL FORECASTS COMPLETE!")
print("="*80)
print("\nKey difference from BB model:")
print("  - Shortage is now ENDOGENOUS (depends on excess demand and GSCPI)")
print("  - Wage equation includes capacity utilization")
print("\n")

# %%
