# %%
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
What Caused the U.S Pandemic-Era Inflation? Bernanke, Blanchard 2023
Python replication of cond_forecast.m
This version: December 3, 2024

This script runs the conditional forecast from Bernanke and Blanchard (2023).

We run conditional forecasts using three assumptions:
1. v/u declines to 0.8 after 8 quarters (tight labor market normalization)
2. v/u declines to 1.2 after 8 quarters (target steady state)
3. v/u remains about steady at 1.8 (persistent tight labor market)

Key parameters:
- vu_decline_val: terminal value of v/u
- vu_quarters_decline: number of quarters over which v/u declines
- years: forecast horizon in years (default 100)

The script also calculates a wage constant adjustment for steady state targeting.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from dateutil.relativedelta import relativedelta

# %%

#****************************CHANGE PATH HERE************************************
# Input Location - coefficients and data from regression_full.py
coef_path = Path("/Users/johnathansun/Documents/ec1499/Replication Package/Code and Data/(2) Regressions/Output Data Python (Full Sample)/eq_coefficients_python.xlsx")
data_path = Path("/Users/johnathansun/Documents/ec1499/Replication Package/Code and Data/(2) Regressions/Output Data Python (Full Sample)/eq_simulations_data_python.xlsx")

# Output Location
output_dir = Path("/Users/johnathansun/Documents/ec1499/Replication Package/Code and Data/(3) Core Results/Conditional Forecasts/Output Data Python")
output_dir.mkdir(parents=True, exist_ok=True)

#*********************************************************************************

print("Loading data and coefficients...")

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
    """Load coefficients from Excel file"""
    gw_beta = pd.read_excel(coef_path, sheet_name='gw')
    gcpi_beta = pd.read_excel(coef_path, sheet_name='gcpi')
    cf1_beta = pd.read_excel(coef_path, sheet_name='cf1')
    cf10_beta = pd.read_excel(coef_path, sheet_name='cf10')

    return {
        'gw': gw_beta['beta'].values,
        'gcpi': gcpi_beta['beta'].values,
        'cf1': cf1_beta['beta'].values,
        'cf10': cf10_beta['beta'].values
    }


def calculate_wage_constant(gw_beta, gcpi_beta, magpty_star, shortage_star, vu_star):
    """
    Calculate the wage constant adjustment for steady state targeting.

    This ensures the model converges to the desired steady state values.
    See wage_constant.pdf in the replication package for derivation.
    """
    # Sum of vu coefficients (indices 10-13 in gw equation, after constant)
    sum_vu_coef = sum(gw_beta[10:14])

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
        (1 - sum_gw_lag_coef) * (
            (magpty_coef_gcpi / (1 - sum_gcpi_lag_coef) +
             magpty_coef_gw / (1 - sum_gw_lag_coef)) * magpty_star +
            sum_shortage_coef / (1 - sum_gcpi_lag_coef) * shortage_star +
            gcpi_const / (1 - sum_gcpi_lag_coef)
        )
    )

    return wage_constant


def conditional_forecast(data, coef_path, wage_constant, magpty_star, shortage_star,
                         vu_decline_val, vu_quarters_decline, years=100):
    """
    Run conditional forecast with specified v/u path.

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
    shortage_star : float
        Long-run shortage index target
    vu_decline_val : float
        Terminal value of v/u ratio
    vu_quarters_decline : int
        Number of quarters for v/u to reach terminal value
    years : int
        Forecast horizon in years

    Returns:
    --------
    out_data : DataFrame
        Forecast results
    """

    # Load coefficients
    coeffs = load_coefficients(coef_path)
    gw_beta = coeffs['gw']
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

    # Initialize simulation arrays (first 4 values from historical data)
    gw_simul = np.zeros(timesteps)
    gcpi_simul = np.zeros(timesteps)
    cf1_simul = np.zeros(timesteps)
    cf10_simul = np.zeros(timesteps)
    diffcpicf_simul = np.zeros(timesteps)

    gw_simul[:4] = gw[:4]
    gcpi_simul[:4] = gcpi[:4]
    cf1_simul[:4] = cf1[:4]
    cf10_simul[:4] = cf10[:4]
    diffcpicf_simul[:4] = diffcpicf[:4]

    # Initialize exogenous variables
    grpe_simul = np.zeros(timesteps)
    grpf_simul = np.zeros(timesteps)
    vu_simul = np.zeros(timesteps)
    shortage_simul = np.zeros(timesteps)
    magpty_simul = np.zeros(timesteps)

    grpe_simul[:len(grpe)] = grpe
    grpf_simul[:len(grpf)] = grpf
    vu_simul[:len(vu)] = vu
    shortage_simul[:len(shortage)] = shortage
    magpty_simul[:len(magpty)] = magpty

    # Calculate slope for v/u transition
    vu_slope = (vu_decline_val - vu_simul[3]) / vu_quarters_decline

    # Run forecast
    for t in range(4, timesteps):

        # Set exogenous variables to steady state values
        grpe_simul[t] = 0
        grpf_simul[t] = 0
        shortage_simul[t] = shortage_star
        magpty_simul[t] = magpty_star

        # V/U path: gradual transition to terminal value
        if vu_slope >= 0:
            vu_simul[t] = min(vu_decline_val, vu_simul[t-1] + vu_slope)
        else:
            vu_simul[t] = max(vu_decline_val, vu_simul[t-1] + vu_slope)

        # Wage equation with adjusted constant
        # Note: We use wage_constant instead of dummies and original constant
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
            wage_constant
        )

        # Price equation
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
            gcpi_beta[21] * shortage_simul[t] +
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
        'shortage_simul': shortage_simul
    })

    return out_data


# %%
print("\n" + "="*80)
print("CALCULATING WAGE CONSTANT")
print("="*80)

# Load coefficients for wage constant calculation
coeffs = load_coefficients(coef_path)
gw_beta = coeffs['gw']
gcpi_beta = coeffs['gcpi']

# Steady state parameters
magpty_star = 1.0      # Long-run productivity growth
shortage_star = 10.0   # Long-run shortage index
vu_star = 1.2          # Long-run target v/u

wage_constant = calculate_wage_constant(gw_beta, gcpi_beta, magpty_star, shortage_star, vu_star)
print(f"Wage constant adjustment: {wage_constant:.6f}")


# %%
print("\n" + "="*80)
print("RUNNING CONDITIONAL FORECASTS")
print("="*80)

# Scenario 1: v/u declines to 0.8 (tight labor market normalizes significantly)
print("\nScenario 1: v/u -> 0.8")
result_low = conditional_forecast(
    data, coef_path, wage_constant, magpty_star, shortage_star,
    vu_decline_val=0.8,
    vu_quarters_decline=8,
    years=100
)

# Scenario 2: v/u declines to 1.2 (target steady state)
print("\nScenario 2: v/u -> 1.2 (target)")
result_mid = conditional_forecast(
    data, coef_path, wage_constant, magpty_star, shortage_star,
    vu_decline_val=1.2,
    vu_quarters_decline=8,
    years=100
)

# Scenario 3: v/u stays at 1.8 (persistent tight labor market)
print("\nScenario 3: v/u -> 1.8")
result_high = conditional_forecast(
    data, coef_path, wage_constant, magpty_star, shortage_star,
    vu_decline_val=1.8,
    vu_quarters_decline=8,
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
print("FORECAST SUMMARY")
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

print("\n" + "="*80)
print("CONDITIONAL FORECASTS COMPLETE!")
print("="*80)
print("\n")

# %%
