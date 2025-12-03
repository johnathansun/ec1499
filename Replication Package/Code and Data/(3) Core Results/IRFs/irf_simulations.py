# %%
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
What Caused the U.S Pandemic-Era Inflation? Bernanke, Blanchard 2023
Python replication of irf_simulations.m
This version: December 3, 2024

This file generates figures 10 and 11 from Bernanke and Blanchard (2023).

The script:
1. Reads estimated coefficients from regression output
2. Simulates impulse response functions for various shocks
3. Outputs results to Excel files for plotting

Shocks analyzed:
- Energy prices (grpe) - one-time shock
- Food prices (grpf) - one-time shock
- V/U ratio - persistent shock (rho=1.0)
- Shortages - one-time shock
"""

import pandas as pd
import numpy as np
from pathlib import Path

# %%

#****************************CHANGE PATH HERE************************************
# Input Location - coefficients and data from regression_full.py
# coef_path = Path("../../(2) Regressions/Output Data (Full Sample)/eq_coefficients.xlsx")
# data_path = Path("../../(2) Regressions/Output Data (Full Sample)/eq_simulations_data.xls")
coef_path = Path("/Users/johnathansun/Documents/ec1499/Replication Package/Code and Data/(2) Regressions/Output Data Python (Full Sample)/eq_coefficients_python.xlsx")
data_path = Path("/Users/johnathansun/Documents/ec1499/Replication Package/Code and Data/(2) Regressions/Output Data Python (Full Sample)/eq_simulations_data_python.xlsx")

# Output Location
output_dir = Path("/Users/johnathansun/Documents/ec1499/Replication Package/Code and Data/(3) Core Results/IRFs/Output Data Python")
output_dir.mkdir(parents=True, exist_ok=True)

#*********************************************************************************

print("Loading data and coefficients...")

# Load simulation data
data = pd.read_excel(data_path)

# Convert period to datetime if needed
if not pd.api.types.is_datetime64_any_dtype(data['period']):
    data['period'] = pd.to_datetime(data['period'])

# Filter data for Q4 2019 onwards (for calculating shock standard deviations)
table_q4_data = data[data['period'] >= '2020-01-01'].copy()

# Filter data from Q4 2018 onwards (for reference, though IRFs start from steady state)
data = data[data['period'] >= '2018-10-01'].copy()

print(f"Data loaded: {len(data)} observations")
print(f"Q4+ data for shock calculation: {len(table_q4_data)} observations")


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


def irfs(data, table_q4_data, shocks, rho, coef_path):
    """
    Run impulse response functions with specified shocks.

    Parameters:
    -----------
    data : DataFrame
        Historical data (used for reference)
    table_q4_data : DataFrame
        Data from Q4 2019+ for calculating shock magnitudes
    shocks : list of bool
        [grpe, grpf, vu, shortage, gw, gcpi, cf1, cf10] - which shocks to apply
    rho : list of float
        Persistence parameters for each shock
    coef_path : Path
        Path to coefficients Excel file

    Returns:
    --------
    results : DataFrame
        IRF simulation results
    """

    # Load coefficients
    coeffs = load_coefficients(coef_path)
    gw_beta = coeffs['gw']
    gcpi_beta = coeffs['gcpi']
    cf1_beta = coeffs['cf1']
    cf10_beta = coeffs['cf10']

    # Initialize persistence parameters
    rho_grpe, rho_grpf, rho_vu, rho_shortage = rho[0], rho[1], rho[2], rho[3]
    rho_gw, rho_gcpi, rho_cf1, rho_cf10 = rho[4], rho[5], rho[6], rho[7]

    # Initialize shock flags
    add_grpe_shock = shocks[0]
    add_grpf_shock = shocks[1]
    add_vu_shock = shocks[2]
    add_shortage_shock = shocks[3]
    add_gw_shock = shocks[4]
    add_gcpi_shock = shocks[5]
    add_cf1_shock = shocks[6]
    add_cf10_shock = shocks[7]

    # Track which shocks are added
    shocks_added = []
    if add_grpe_shock: shocks_added.append("grpe")
    if add_grpf_shock: shocks_added.append("grpf")
    if add_vu_shock: shocks_added.append("vu")
    if add_shortage_shock: shocks_added.append("shortage")
    if add_gw_shock: shocks_added.append("gw")
    if add_gcpi_shock: shocks_added.append("gcpi")
    if add_cf1_shock: shocks_added.append("cf1")
    if add_cf10_shock: shocks_added.append("cf10")

    print(f"  Running IRF with shocks: {shocks_added}")

    # Define time horizon
    timesteps = 32

    # Initialize empty arrays (all zeros - steady state)
    gw = np.zeros(timesteps)
    cf1 = np.zeros(timesteps)
    magpty = np.zeros(timesteps)
    diffcpicf = np.zeros(timesteps)
    vu = np.zeros(timesteps)
    gcpi = np.zeros(timesteps)
    grpe = np.zeros(timesteps)
    grpf = np.zeros(timesteps)
    shortage = np.zeros(timesteps)
    cf10 = np.zeros(timesteps)

    # Initialize simulation arrays (first 4 values from steady state = 0)
    gw_simul = np.zeros(timesteps)
    gcpi_simul = np.zeros(timesteps)
    cf1_simul = np.zeros(timesteps)
    cf10_simul = np.zeros(timesteps)
    diffcpicf_simul = np.zeros(timesteps)

    # Initialize exogenous shock series
    grpe_shock_series = np.zeros(timesteps)
    grpf_shock_series = np.zeros(timesteps)
    vu_shock_series = np.zeros(timesteps)
    shortage_shock_series = np.zeros(timesteps)

    # Calculate shock values (1 standard deviation)
    shock_val_grpe = np.nanstd(table_q4_data['grpe'])
    shock_val_grpf = np.nanstd(table_q4_data['grpf'])
    shock_val_vu = np.nanstd(table_q4_data['vu'])
    shock_val_shortage = np.nanstd(table_q4_data['shortage'])
    shock_val_gw_residual = np.nanstd(table_q4_data['gw_residuals'])
    shock_val_gcpi_residual = np.nanstd(table_q4_data['gcpi_residuals'])
    shock_val_cf1_residual = np.nanstd(table_q4_data['cf1_residuals'])
    shock_val_cf10_residual = np.nanstd(table_q4_data['cf10_residuals'])

    # Initialize shock arrays
    shock_grpe = np.zeros(timesteps)
    shock_grpf = np.zeros(timesteps)
    shock_vu = np.zeros(timesteps)
    shock_shortage = np.zeros(timesteps)
    shock_gw = np.zeros(timesteps)
    shock_gcpi = np.zeros(timesteps)
    shock_cf1 = np.zeros(timesteps)
    shock_cf10 = np.zeros(timesteps)

    # Run IRF simulation (MATLAB uses 1-indexing, Python uses 0-indexing)
    # In MATLAB, t starts at 5; in Python, t starts at 4
    for t in range(4, timesteps):

        # Reset shocks each period
        shock_grpe[t] = 0
        shock_grpf[t] = 0
        shock_vu[t] = 0
        shock_shortage[t] = 0
        shock_gw[t] = 0
        shock_gcpi[t] = 0
        shock_cf1[t] = 0
        shock_cf10[t] = 0

        # Apply shocks at t=4 (first simulation period, equivalent to MATLAB t=5)
        if add_grpe_shock and t == 4:
            shock_grpe[t] = shock_val_grpe

        if add_grpf_shock and t == 4:
            shock_grpf[t] = shock_val_grpf

        if add_vu_shock and t == 4:
            shock_vu[t] = shock_val_vu

        if add_shortage_shock and t == 4:
            shock_shortage[t] = shock_val_shortage

        if add_gw_shock and t == 4:
            shock_gw[t] = shock_val_gw_residual

        if add_gcpi_shock and t == 4:
            shock_gcpi[t] = shock_val_gcpi_residual

        if add_cf1_shock and t == 4:
            shock_cf1[t] = shock_val_cf1_residual

        if add_cf10_shock and t == 4:
            shock_cf10[t] = shock_val_cf10_residual

        # Update exogenous shock series with persistence
        grpe_shock_series[t] = rho_grpe * grpe_shock_series[t-1] + shock_grpe[t]
        grpf_shock_series[t] = rho_grpf * grpf_shock_series[t-1] + shock_grpf[t]
        vu_shock_series[t] = rho_vu * vu_shock_series[t-1] + shock_vu[t]
        shortage_shock_series[t] = rho_shortage * shortage_shock_series[t-1] + shock_shortage[t]

        # Wage equation
        # Coefficients: [const, L1-L4 gw, L1-L4 cf1, L1 magpty, L1-L4 vu, L1-L4 diffcpicf, dummies...]
        # For full sample: indices 1-4 (gw), 5-8 (cf1), 9 (magpty), 10-13 (vu), 14-17 (diffcpicf)
        if add_gw_shock and t == 4:
            gw_simul[t] = rho_gw * gw_simul[t-1] + shock_gw[t]
        else:
            gw_simul[t] = (
                gw_beta[1] * gw_simul[t-1] +
                gw_beta[2] * gw_simul[t-2] +
                gw_beta[3] * gw_simul[t-3] +
                gw_beta[4] * gw_simul[t-4] +
                gw_beta[5] * cf1_simul[t-1] +
                gw_beta[6] * cf1_simul[t-2] +
                gw_beta[7] * cf1_simul[t-3] +
                gw_beta[8] * cf1_simul[t-4] +
                gw_beta[9] * magpty[t-1] +
                gw_beta[10] * vu_shock_series[t-1] +
                gw_beta[11] * vu_shock_series[t-2] +
                gw_beta[12] * vu_shock_series[t-3] +
                gw_beta[13] * vu_shock_series[t-4] +
                gw_beta[14] * diffcpicf_simul[t-1] +
                gw_beta[15] * diffcpicf_simul[t-2] +
                gw_beta[16] * diffcpicf_simul[t-3] +
                gw_beta[17] * diffcpicf_simul[t-4]
            )

        # Price equation
        # Coefficients: [const, magpty, L1-L4 gcpi, gw L1-L4 gw, grpe L1-L4 grpe, grpf L1-L4 grpf, shortage L1-L4 shortage]
        if add_gcpi_shock and t == 4:
            gcpi_simul[t] = rho_gcpi * gcpi_simul[t-1] + shock_gcpi[t]
        else:
            gcpi_simul[t] = (
                gcpi_beta[1] * magpty[t] +
                gcpi_beta[2] * gcpi_simul[t-1] +
                gcpi_beta[3] * gcpi_simul[t-2] +
                gcpi_beta[4] * gcpi_simul[t-3] +
                gcpi_beta[5] * gcpi_simul[t-4] +
                gcpi_beta[6] * gw_simul[t] +
                gcpi_beta[7] * gw_simul[t-1] +
                gcpi_beta[8] * gw_simul[t-2] +
                gcpi_beta[9] * gw_simul[t-3] +
                gcpi_beta[10] * gw_simul[t-4] +
                gcpi_beta[11] * grpe_shock_series[t] +
                gcpi_beta[12] * grpe_shock_series[t-1] +
                gcpi_beta[13] * grpe_shock_series[t-2] +
                gcpi_beta[14] * grpe_shock_series[t-3] +
                gcpi_beta[15] * grpe_shock_series[t-4] +
                gcpi_beta[16] * grpf_shock_series[t] +
                gcpi_beta[17] * grpf_shock_series[t-1] +
                gcpi_beta[18] * grpf_shock_series[t-2] +
                gcpi_beta[19] * grpf_shock_series[t-3] +
                gcpi_beta[20] * grpf_shock_series[t-4] +
                gcpi_beta[21] * shortage_shock_series[t] +
                gcpi_beta[22] * shortage_shock_series[t-1] +
                gcpi_beta[23] * shortage_shock_series[t-2] +
                gcpi_beta[24] * shortage_shock_series[t-3] +
                gcpi_beta[25] * shortage_shock_series[t-4]
            )

        # Catch-up term
        diffcpicf_simul[t] = 0.25 * (gcpi_simul[t] + gcpi_simul[t-1] +
                                     gcpi_simul[t-2] + gcpi_simul[t-3]) - cf1_simul[t-4]

        # Long-run expectations (cf10)
        # Coefficients: [L1-L4 cf10, gcpi L1-L4 gcpi] (no constant)
        if add_cf10_shock and t == 4:
            cf10_simul[t] = rho_cf10 * cf10_simul[t-1] + shock_cf10[t]
        else:
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
        # Coefficients: [L1-L4 cf1, cf10 L1-L4 cf10, gcpi L1-L4 gcpi] (no constant)
        if add_cf1_shock and t == 4:
            cf1_simul[t] = rho_cf1 * cf1_simul[t-1] + shock_cf1[t]
        else:
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

    # Create results DataFrame (1-indexed period to match MATLAB)
    results = pd.DataFrame({
        'period': np.arange(1, timesteps + 1),
        'gw_simul': gw_simul,
        'gcpi_simul': gcpi_simul,
        'cf1_simul': cf1_simul,
        'cf10_simul': cf10_simul,
        'diffcpicf_simul': diffcpicf_simul,
        'grpe_shock_series': grpe_shock_series,
        'grpf_shock_series': grpf_shock_series,
        'vu_shock_series': vu_shock_series,
        'shortage_shock_series': shortage_shock_series
    })

    return results


# %%
print("\n" + "="*80)
print("RUNNING IRF SIMULATIONS")
print("="*80)

# Figure 10: Energy prices shock
print("\nFigure 10 - Energy prices shock:")
results_energy = irfs(
    data, table_q4_data,
    shocks=[True, False, False, False, False, False, False, False],  # grpe only
    rho=[0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    coef_path=coef_path
)

# Figure 10: Food prices shock
print("\nFigure 10 - Food prices shock:")
results_food = irfs(
    data, table_q4_data,
    shocks=[False, True, False, False, False, False, False, False],  # grpf only
    rho=[0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    coef_path=coef_path
)

# Figure 11: V/U shock (persistent)
print("\nFigure 11 - V/U shock:")
results_vu = irfs(
    data, table_q4_data,
    shocks=[False, False, True, False, False, False, False, False],  # vu only
    rho=[0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # rho_vu = 1.0 (permanent)
    coef_path=coef_path
)

# Figure 10: Shortage shock
print("\nFigure 10 - Shortage shock:")
results_shortage = irfs(
    data, table_q4_data,
    shocks=[False, False, False, True, False, False, False, False],  # shortage only
    rho=[0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    coef_path=coef_path
)


# %%
print("\n" + "="*80)
print("EXPORTING RESULTS")
print("="*80)

# Export to Excel
results_energy.to_excel(output_dir / 'results_energy.xlsx', index=False)
results_food.to_excel(output_dir / 'results_food.xlsx', index=False)
results_vu.to_excel(output_dir / 'results_vu.xlsx', index=False)
results_shortage.to_excel(output_dir / 'results_shortage.xlsx', index=False)

print(f"\nResults saved to: {output_dir}")
print("  - results_energy.xlsx")
print("  - results_food.xlsx")
print("  - results_vu.xlsx")
print("  - results_shortage.xlsx")


# %%
# Print summary of IRF responses
print("\n" + "="*80)
print("IRF SUMMARY (Peak inflation response)")
print("="*80)

print(f"\nEnergy shock:   Peak gcpi = {results_energy['gcpi_simul'].max():.4f} at period {results_energy['gcpi_simul'].argmax() + 1}")
print(f"Food shock:     Peak gcpi = {results_food['gcpi_simul'].max():.4f} at period {results_food['gcpi_simul'].argmax() + 1}")
print(f"V/U shock:      Peak gcpi = {results_vu['gcpi_simul'].max():.4f} at period {results_vu['gcpi_simul'].argmax() + 1}")
print(f"Shortage shock: Peak gcpi = {results_shortage['gcpi_simul'].max():.4f} at period {results_shortage['gcpi_simul'].argmax() + 1}")

print("\n" + "="*80)
print("IRF SIMULATIONS COMPLETE!")
print("="*80)
print("\n")

# %%
