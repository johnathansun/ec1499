# %%
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
What Caused the U.S Pandemic-Era Inflation? Bernanke, Blanchard 2023
Python replication of decomp.m
This version: December 3, 2024

This file generates the output to create figures 12 and 13 from Bernanke and Blanchard (2023).

The script:
1. Runs a baseline simulation with all exogenous shocks included
2. Runs counterfactual simulations removing one shock at a time
3. Calculates the contribution of each shock: baseline - counterfactual
4. Outputs results to Excel files for plotting

Shocks analyzed:
- Energy prices (grpe)
- Food prices (grpf)
- V/U ratio
- Shortages
- Productivity (magpty)
- Q2 2020 dummy
- Q3 2020 dummy
"""

import pandas as pd
import numpy as np
from pathlib import Path

# %%

#****************************CHANGE PATH HERE************************************
# Input Location - coefficients and data from regression_full.py
coef_path = Path("/Users/johnathansun/Documents/ec1499/Replication Package/Code and Data/(2) Regressions/Output Data Python (Full Sample)/eq_coefficients_python.xlsx")
data_path = Path("/Users/johnathansun/Documents/ec1499/Replication Package/Code and Data/(2) Regressions/Output Data Python (Full Sample)/eq_simulations_data_python.xlsx")

# Output Location
output_dir = Path("/Users/johnathansun/Documents/ec1499/Replication Package/Code and Data/(3) Core Results/Decompositions/Output Data Python")
output_dir.mkdir(parents=True, exist_ok=True)

#*********************************************************************************

print("Loading data and coefficients...")

# Load simulation data
data = pd.read_excel(data_path)

# Convert period to datetime if needed
if not pd.api.types.is_datetime64_any_dtype(data['period']):
    data['period'] = pd.to_datetime(data['period'])

# Filter data from Q4 2018 onwards
data = data[data['period'] >= '2018-10-01'].copy().reset_index(drop=True)

print(f"Data loaded: {len(data)} observations")
print(f"Period: {data['period'].min()} to {data['period'].max()}")


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


def dynamic_simul(data, coef_path, remove_grpe=False, remove_grpf=False,
                  remove_vu=False, remove_shortage=False, remove_magpty=False,
                  remove_dummy2020_q2=False, remove_dummy2020_q3=False):
    """
    Run dynamic simulation with specified shocks removed.

    Parameters:
    -----------
    data : DataFrame
        Historical data
    coef_path : Path
        Path to coefficients Excel file
    remove_* : bool
        Whether to remove each shock from the simulation

    Returns:
    --------
    out_data : DataFrame
        Simulation results with contributions
    """

    # Load coefficients
    coeffs = load_coefficients(coef_path)
    gw_beta = coeffs['gw']
    gcpi_beta = coeffs['gcpi']
    cf1_beta = coeffs['cf1']
    cf10_beta = coeffs['cf10']

    # Track which shocks are removed
    shocks_removed = []
    if remove_grpe: shocks_removed.append("grpe")
    if remove_grpf: shocks_removed.append("grpf")
    if remove_vu: shocks_removed.append("vu")
    if remove_shortage: shocks_removed.append("shortage")
    if remove_magpty: shocks_removed.append("magpty")
    if remove_dummy2020_q2: shocks_removed.append("dummy2020_q2")
    if remove_dummy2020_q3: shocks_removed.append("dummy2020_q3")

    print(f"  Running simulation with shocks removed: {shocks_removed if shocks_removed else 'None (baseline)'}")

    # Extract historical data as arrays
    timesteps = len(data)
    period = data['period'].values

    # Endogenous variables (historical)
    gw = data['gw'].values.copy()
    gcpi = data['gcpi'].values.copy()
    cf1 = data['cf1'].values.copy()
    cf10 = data['cf10'].values.copy()
    diffcpicf = data['diffcpicf'].values.copy()

    # Exogenous variables (historical)
    grpe = data['grpe'].values.copy()
    grpf = data['grpf'].values.copy()
    vu = data['vu'].values.copy()
    shortage = data['shortage'].values.copy()
    magpty = data['magpty'].values.copy()

    # Define dummy variables (Q2 2020 is index 6, Q3 2020 is index 7 in 0-indexed array starting from Q4 2018)
    dummy_q2 = np.zeros(timesteps)
    dummy_q2[6] = 1.0  # Q2 2020

    dummy_q3 = np.zeros(timesteps)
    dummy_q3[7] = 1.0  # Q3 2020

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

    # Initialize shock series for simulation (can be modified)
    grpe_simul = grpe.copy()
    grpf_simul = grpf.copy()
    vu_simul = vu.copy()
    shortage_simul = shortage.copy()
    magpty_simul = magpty.copy()
    dummy_q2_simul = dummy_q2.copy()
    dummy_q3_simul = dummy_q3.copy()

    # Run simulation with shocks removed as specified
    for t in range(4, timesteps):

        # Set exogenous variables based on removal flags
        if remove_grpe:
            grpe_simul[t] = 0

        if remove_grpf:
            grpf_simul[t] = 0

        if remove_vu:
            # Set to long-run constant (value at t=4, i.e., Q4 2019)
            vu_simul[t] = data['vu'].iloc[4]

        if remove_shortage:
            # Set to long-run constant
            shortage_simul[t] = 5

        if remove_magpty:
            # Set to long-run constant
            magpty_simul[t] = 2

        if remove_dummy2020_q2:
            dummy_q2_simul[t] = 0.0

        if remove_dummy2020_q3:
            dummy_q3_simul[t] = 0.0

        # Wage equation simulation
        # Coefficients: [const, L1-L4 gw, L1-L4 cf1, L1 magpty, L1-L4 vu, L1-L4 diffcpicf, dummy_q2, dummy_q3]
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
            gw_beta[18] * dummy_q2_simul[t] +
            gw_beta[19] * dummy_q3_simul[t] +
            gw_beta[0]  # constant
        )

        # Price equation simulation
        # Coefficients: [const, magpty, L1-L4 gcpi, gw L1-L4 gw, grpe L1-L4 grpe, grpf L1-L4 grpf, shortage L1-L4 shortage]
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
            gcpi_beta[0]  # constant
        )

        # Catch-up term
        diffcpicf_simul[t] = 0.25 * (gcpi_simul[t] + gcpi_simul[t-1] +
                                     gcpi_simul[t-2] + gcpi_simul[t-3]) - cf1_simul[t-4]

        # Long-run expectations (cf10) - no constant
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

        # Short-run expectations (cf1) - no constant
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

    # Run baseline simulation (all shocks included)
    gw_simul_baseline = np.zeros(timesteps)
    gcpi_simul_baseline = np.zeros(timesteps)
    cf1_simul_baseline = np.zeros(timesteps)
    cf10_simul_baseline = np.zeros(timesteps)
    diffcpicf_simul_baseline = np.zeros(timesteps)

    gw_simul_baseline[:4] = gw[:4]
    gcpi_simul_baseline[:4] = gcpi[:4]
    cf1_simul_baseline[:4] = cf1[:4]
    cf10_simul_baseline[:4] = cf10[:4]
    diffcpicf_simul_baseline[:4] = diffcpicf[:4]

    for t in range(4, timesteps):
        # Wage equation (baseline with all shocks)
        gw_simul_baseline[t] = (
            gw_beta[1] * gw_simul_baseline[t-1] +
            gw_beta[2] * gw_simul_baseline[t-2] +
            gw_beta[3] * gw_simul_baseline[t-3] +
            gw_beta[4] * gw_simul_baseline[t-4] +
            gw_beta[5] * cf1_simul_baseline[t-1] +
            gw_beta[6] * cf1_simul_baseline[t-2] +
            gw_beta[7] * cf1_simul_baseline[t-3] +
            gw_beta[8] * cf1_simul_baseline[t-4] +
            gw_beta[9] * magpty[t-1] +
            gw_beta[10] * vu[t-1] +
            gw_beta[11] * vu[t-2] +
            gw_beta[12] * vu[t-3] +
            gw_beta[13] * vu[t-4] +
            gw_beta[14] * diffcpicf_simul_baseline[t-1] +
            gw_beta[15] * diffcpicf_simul_baseline[t-2] +
            gw_beta[16] * diffcpicf_simul_baseline[t-3] +
            gw_beta[17] * diffcpicf_simul_baseline[t-4] +
            gw_beta[18] * dummy_q2[t] +
            gw_beta[19] * dummy_q3[t] +
            gw_beta[0]
        )

        # Price equation (baseline)
        gcpi_simul_baseline[t] = (
            gcpi_beta[1] * magpty[t] +
            gcpi_beta[2] * gcpi_simul_baseline[t-1] +
            gcpi_beta[3] * gcpi_simul_baseline[t-2] +
            gcpi_beta[4] * gcpi_simul_baseline[t-3] +
            gcpi_beta[5] * gcpi_simul_baseline[t-4] +
            gcpi_beta[6] * gw_simul_baseline[t] +
            gcpi_beta[7] * gw_simul_baseline[t-1] +
            gcpi_beta[8] * gw_simul_baseline[t-2] +
            gcpi_beta[9] * gw_simul_baseline[t-3] +
            gcpi_beta[10] * gw_simul_baseline[t-4] +
            gcpi_beta[11] * grpe[t] +
            gcpi_beta[12] * grpe[t-1] +
            gcpi_beta[13] * grpe[t-2] +
            gcpi_beta[14] * grpe[t-3] +
            gcpi_beta[15] * grpe[t-4] +
            gcpi_beta[16] * grpf[t] +
            gcpi_beta[17] * grpf[t-1] +
            gcpi_beta[18] * grpf[t-2] +
            gcpi_beta[19] * grpf[t-3] +
            gcpi_beta[20] * grpf[t-4] +
            gcpi_beta[21] * shortage[t] +
            gcpi_beta[22] * shortage[t-1] +
            gcpi_beta[23] * shortage[t-2] +
            gcpi_beta[24] * shortage[t-3] +
            gcpi_beta[25] * shortage[t-4] +
            gcpi_beta[0]
        )

        # Catch-up term (baseline)
        diffcpicf_simul_baseline[t] = 0.25 * (gcpi_simul_baseline[t] + gcpi_simul_baseline[t-1] +
                                               gcpi_simul_baseline[t-2] + gcpi_simul_baseline[t-3]) - cf1_simul_baseline[t-4]

        # Long-run expectations (baseline)
        cf10_simul_baseline[t] = (
            cf10_beta[0] * cf10_simul_baseline[t-1] +
            cf10_beta[1] * cf10_simul_baseline[t-2] +
            cf10_beta[2] * cf10_simul_baseline[t-3] +
            cf10_beta[3] * cf10_simul_baseline[t-4] +
            cf10_beta[4] * gcpi_simul_baseline[t] +
            cf10_beta[5] * gcpi_simul_baseline[t-1] +
            cf10_beta[6] * gcpi_simul_baseline[t-2] +
            cf10_beta[7] * gcpi_simul_baseline[t-3] +
            cf10_beta[8] * gcpi_simul_baseline[t-4]
        )

        # Short-run expectations (baseline)
        cf1_simul_baseline[t] = (
            cf1_beta[0] * cf1_simul_baseline[t-1] +
            cf1_beta[1] * cf1_simul_baseline[t-2] +
            cf1_beta[2] * cf1_simul_baseline[t-3] +
            cf1_beta[3] * cf1_simul_baseline[t-4] +
            cf1_beta[4] * cf10_simul_baseline[t] +
            cf1_beta[5] * cf10_simul_baseline[t-1] +
            cf1_beta[6] * cf10_simul_baseline[t-2] +
            cf1_beta[7] * cf10_simul_baseline[t-3] +
            cf1_beta[8] * cf10_simul_baseline[t-4] +
            cf1_beta[9] * gcpi_simul_baseline[t] +
            cf1_beta[10] * gcpi_simul_baseline[t-1] +
            cf1_beta[11] * gcpi_simul_baseline[t-2] +
            cf1_beta[12] * gcpi_simul_baseline[t-3] +
            cf1_beta[13] * gcpi_simul_baseline[t-4]
        )

    # Create output DataFrame
    out_data = pd.DataFrame({
        'period': period,
        'gw': gw,
        'gw_simul': gw_simul,
        'gw_simul_baseline': gw_simul_baseline,
        'gcpi': gcpi,
        'gcpi_simul': gcpi_simul,
        'gcpi_simul_baseline': gcpi_simul_baseline,
        'cf1': cf1,
        'cf1_simul': cf1_simul,
        'cf1_simul_baseline': cf1_simul_baseline,
        'cf10': cf10,
        'cf10_simul': cf10_simul,
        'cf10_simul_baseline': cf10_simul_baseline,
        'grpe': grpe,
        'grpe_simul': grpe_simul,
        'grpf': grpf,
        'grpf_simul': grpf_simul,
        'vu': vu,
        'vu_simul': vu_simul,
        'shortage': shortage,
        'shortage_simul': shortage_simul,
        'magpty': magpty,
        'magpty_simul': magpty_simul
    })

    # Calculate contributions if only one shock is removed
    if len(shocks_removed) == 1:
        shock_name = shocks_removed[0]
        contr_gw = gw_simul_baseline - gw_simul
        contr_gcpi = gcpi_simul_baseline - gcpi_simul
        contr_cf1 = cf1_simul_baseline - cf1_simul
        contr_cf10 = cf10_simul_baseline - cf10_simul

        out_data[f'{shock_name}_contr_gw'] = contr_gw
        out_data[f'{shock_name}_contr_gcpi'] = contr_gcpi
        out_data[f'{shock_name}_contr_cf1'] = contr_cf1
        out_data[f'{shock_name}_contr_cf10'] = contr_cf10

    return out_data


# %%
print("\n" + "="*80)
print("RUNNING DECOMPOSITION SIMULATIONS")
print("="*80)

# Run all decomposition scenarios
print("\nBaseline (remove all shocks):")
remove_all = dynamic_simul(data, coef_path,
                           remove_grpe=True, remove_grpf=True, remove_vu=True,
                           remove_shortage=True, remove_magpty=True,
                           remove_dummy2020_q2=True, remove_dummy2020_q3=True)

print("\nRemove energy prices only:")
remove_grpe = dynamic_simul(data, coef_path, remove_grpe=True)

print("\nRemove food prices only:")
remove_grpf = dynamic_simul(data, coef_path, remove_grpf=True)

print("\nRemove V/U only:")
remove_vu = dynamic_simul(data, coef_path, remove_vu=True)

print("\nRemove shortage only:")
remove_shortage = dynamic_simul(data, coef_path, remove_shortage=True)

print("\nRemove productivity only:")
remove_magpty = dynamic_simul(data, coef_path, remove_magpty=True)

print("\nRemove Q2 2020 dummy only:")
remove_dummy2020_q2 = dynamic_simul(data, coef_path, remove_dummy2020_q2=True)

print("\nRemove Q3 2020 dummy only:")
remove_dummy2020_q3 = dynamic_simul(data, coef_path, remove_dummy2020_q3=True)


# %%
print("\n" + "="*80)
print("EXPORTING RESULTS")
print("="*80)

# Export to Excel
remove_all.to_excel(output_dir / 'remove_all.xlsx', index=False)
remove_grpe.to_excel(output_dir / 'remove_grpe.xlsx', index=False)
remove_grpf.to_excel(output_dir / 'remove_grpf.xlsx', index=False)
remove_vu.to_excel(output_dir / 'remove_vu.xlsx', index=False)
remove_shortage.to_excel(output_dir / 'remove_shortage.xlsx', index=False)
remove_magpty.to_excel(output_dir / 'remove_magpty.xlsx', index=False)
remove_dummy2020_q2.to_excel(output_dir / 'remove_2020q2.xlsx', index=False)
remove_dummy2020_q3.to_excel(output_dir / 'remove_2020q3.xlsx', index=False)

print(f"\nResults saved to: {output_dir}")
print("  - remove_all.xlsx")
print("  - remove_grpe.xlsx")
print("  - remove_grpf.xlsx")
print("  - remove_vu.xlsx")
print("  - remove_shortage.xlsx")
print("  - remove_magpty.xlsx")
print("  - remove_2020q2.xlsx")
print("  - remove_2020q3.xlsx")


# %%
# Print summary
print("\n" + "="*80)
print("DECOMPOSITION SUMMARY")
print("="*80)

# Filter to 2020+ for summary
summary_mask = remove_all['period'] >= '2020-01-01'

print(f"\nPeak contributions to inflation (gcpi) from 2020 onwards:")
print(f"  Energy:      {remove_grpe.loc[summary_mask, 'grpe_contr_gcpi'].max():.2f}")
print(f"  Food:        {remove_grpf.loc[summary_mask, 'grpf_contr_gcpi'].max():.2f}")
print(f"  V/U:         {remove_vu.loc[summary_mask, 'vu_contr_gcpi'].max():.2f}")
print(f"  Shortage:    {remove_shortage.loc[summary_mask, 'shortage_contr_gcpi'].max():.2f}")
print(f"  Productivity:{remove_magpty.loc[summary_mask, 'magpty_contr_gcpi'].max():.2f}")

print("\n" + "="*80)
print("DECOMPOSITION SIMULATIONS COMPLETE!")
print("="*80)
print("\n")

# %%
