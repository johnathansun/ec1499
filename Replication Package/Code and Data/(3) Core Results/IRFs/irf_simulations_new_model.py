# %%
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modified Bernanke-Blanchard Model - Impulse Response Functions
Liang & Sun (2025)

This script generates IRFs for the modified model.

Key differences from original:
1. Wage equation includes capacity utilization (cu)
2. Shortage is ENDOGENOUS: shortage = f(lagged_shortage, excess_demand, GSCPI)
3. Excess demand is ENDOGENOUS: ed = log(W) - log(NGDPPOT) - log(TCU) - trend
   - Wages are computed first
   - Wage level accumulates: log(W_t) = log(W_{t-1}) + gw_t/400
   - Excess demand feeds into shortages
   - Shortages feed into prices

Transmission channels:
- Energy/Food: grpe/grpf → gcpi (direct)
- V/U: vu → gw → W → ed → shortage → gcpi (full feedback loop)
- GSCPI: gscpi → shortage → gcpi
- Capacity utilization: cu → gw → W → ed → shortage → gcpi (full feedback loop)
- Shortage: shortage → gcpi (direct, bypasses shortage equation)

Shocks analyzed:
- Energy prices (grpe) - one-time shock
- Food prices (grpf) - one-time shock
- V/U ratio - persistent shock (rho=1.0)
- Shortage - one-time shock (direct shock to shortage)
- GSCPI - one-time shock [NEW]
- Capacity utilization - one-time shock [NEW]
"""

import pandas as pd
import numpy as np
from pathlib import Path

# %%

#****************************CHANGE PATH HERE************************************
# Input Location - coefficients and data from regression_new_model.py
coef_path = Path("/Users/johnathansun/Documents/ec1499/Replication Package/Code and Data/(2) Regressions/Output Data Python (New Model)/eq_coefficients_new_model.xlsx")
data_path = Path("/Users/johnathansun/Documents/ec1499/Replication Package/Code and Data/(2) Regressions/Output Data Python (New Model)/eq_simulations_data_new_model.xlsx")

# Output Location
output_dir = Path("/Users/johnathansun/Documents/ec1499/Replication Package/Code and Data/(3) Core Results/IRFs/Output Data Python (New Model)")
output_dir.mkdir(parents=True, exist_ok=True)

#*********************************************************************************

print("="*80)
print("MODIFIED MODEL IRF SIMULATIONS")
print("Liang & Sun (2025)")
print("="*80)

print("\nLoading data and coefficients...")

# Load simulation data
data = pd.read_excel(data_path)

# Convert period to datetime if needed
if not pd.api.types.is_datetime64_any_dtype(data['period']):
    data['period'] = pd.to_datetime(data['period'])

# Filter data for Q4 2019 onwards (for calculating shock standard deviations)
table_q4_data = data[data['period'] >= '2020-01-01'].copy()

# Filter data from Q4 2018 onwards
data = data[data['period'] >= '2018-10-01'].copy()

print(f"Data loaded: {len(data)} observations")
print(f"Q4+ data for shock calculation: {len(table_q4_data)} observations")


# %%
def load_coefficients(coef_path):
    """Load coefficients from Excel file (new model with 5 equations)"""
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


def irfs_new_model(data, table_q4_data, shocks, rho, coef_path):
    """
    Run impulse response functions for the modified model.

    IMPORTANT: Excess demand is computed ENDOGENOUSLY from simulated wage levels.
    The simulation order is: Wages → Excess Demand → Shortage → Prices

    Parameters:
    -----------
    data : DataFrame
        Historical data (used for reference)
    table_q4_data : DataFrame
        Data from Q4 2019+ for calculating shock magnitudes
    shocks : dict
        Dictionary of shock flags: {shock_name: True/False}
    rho : dict
        Dictionary of persistence parameters: {shock_name: rho_value}
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
    shortage_beta = coeffs['shortage']
    gcpi_beta = coeffs['gcpi']
    cf1_beta = coeffs['cf1']
    cf10_beta = coeffs['cf10']

    # Extract shock flags
    add_grpe_shock = shocks.get('grpe', False)
    add_grpf_shock = shocks.get('grpf', False)
    add_vu_shock = shocks.get('vu', False)
    add_shortage_shock = shocks.get('shortage', False)
    add_gscpi_shock = shocks.get('gscpi', False)
    add_gcu_shock = shocks.get('gcu', False)

    # Extract persistence parameters
    rho_grpe = rho.get('grpe', 0.0)
    rho_grpf = rho.get('grpf', 0.0)
    rho_vu = rho.get('vu', 1.0)
    rho_shortage = rho.get('shortage', 0.0)
    rho_gscpi = rho.get('gscpi', 0.0)
    rho_gcu = rho.get('gcu', 0.0)

    # Track which shocks are added
    shocks_added = [k for k, v in shocks.items() if v]
    print(f"  Running IRF with shocks: {shocks_added}")

    # Define time horizon
    timesteps = 32

    # Initialize arrays (all zeros - steady state deviations)
    gw_simul = np.zeros(timesteps)
    gcpi_simul = np.zeros(timesteps)
    cf1_simul = np.zeros(timesteps)
    cf10_simul = np.zeros(timesteps)
    shortage_simul = np.zeros(timesteps)
    diffcpicf_simul = np.zeros(timesteps)

    # Level tracking for endogenous excess demand
    # These track DEVIATIONS from steady state path
    log_w_simul = np.zeros(timesteps)  # log wage level deviation
    ed_simul = np.zeros(timesteps)  # excess demand (endogenous)
    raw_ed_simul = np.zeros(timesteps)  # raw excess demand before detrending

    # Exogenous shock series
    grpe_shock_series = np.zeros(timesteps)
    grpf_shock_series = np.zeros(timesteps)
    vu_shock_series = np.zeros(timesteps)
    gscpi_shock_series = np.zeros(timesteps)
    gcu_shock_series = np.zeros(timesteps)

    # Calculate shock values (1 standard deviation)
    shock_val_grpe = np.nanstd(table_q4_data['grpe']) if 'grpe' in table_q4_data.columns else 10.0
    shock_val_grpf = np.nanstd(table_q4_data['grpf']) if 'grpf' in table_q4_data.columns else 5.0
    shock_val_vu = np.nanstd(table_q4_data['vu']) if 'vu' in table_q4_data.columns else 0.3
    shock_val_shortage = np.nanstd(table_q4_data['shortage']) if 'shortage' in table_q4_data.columns else 10.0
    shock_val_gscpi = np.nanstd(table_q4_data['gscpi']) if 'gscpi' in table_q4_data.columns else 1.0
    # Note: 'cu' is the detrended capacity utilization series used in the model
    shock_val_cu = np.nanstd(table_q4_data['cu']) if 'cu' in table_q4_data.columns else 0.04

    print(f"    Shock magnitudes (1 std dev):")
    print(f"      grpe: {shock_val_grpe:.2f}, grpf: {shock_val_grpf:.2f}, vu: {shock_val_vu:.3f}")
    print(f"      shortage: {shock_val_shortage:.2f}, gscpi: {shock_val_gscpi:.2f}, cu: {shock_val_cu:.4f}")

    # Run IRF simulation
    for t in range(4, timesteps):

        # Apply shocks at t=4 (first simulation period)
        shock_grpe = shock_val_grpe if (add_grpe_shock and t == 4) else 0
        shock_grpf = shock_val_grpf if (add_grpf_shock and t == 4) else 0
        shock_vu = shock_val_vu if (add_vu_shock and t == 4) else 0
        shock_gscpi = shock_val_gscpi if (add_gscpi_shock and t == 4) else 0
        shock_cu = shock_val_cu if (add_gcu_shock and t == 4) else 0

        # Update exogenous shock series with persistence
        grpe_shock_series[t] = rho_grpe * grpe_shock_series[t-1] + shock_grpe
        grpf_shock_series[t] = rho_grpf * grpf_shock_series[t-1] + shock_grpf
        vu_shock_series[t] = rho_vu * vu_shock_series[t-1] + shock_vu
        gscpi_shock_series[t] = rho_gscpi * gscpi_shock_series[t-1] + shock_gscpi
        gcu_shock_series[t] = rho_gcu * gcu_shock_series[t-1] + shock_cu

        # =====================================================================
        # STEP 1: WAGE EQUATION (computed FIRST since ED depends on wages)
        # Coefficients: [const, L1-L4 gw, L1-L4 cf1, L1 magpty, L1-L4 vu,
        #                L1-L4 diffcpicf, L1-L4 gcu, dummy_q2, dummy_q3]
        # NOTE: Exclude constant for IRF (measuring deviations from steady state)
        # =====================================================================
        gw_simul[t] = (
            # gw_beta[0] +  # constant EXCLUDED for IRF
            gw_beta[1] * gw_simul[t-1] +
            gw_beta[2] * gw_simul[t-2] +
            gw_beta[3] * gw_simul[t-3] +
            gw_beta[4] * gw_simul[t-4] +
            gw_beta[5] * cf1_simul[t-1] +
            gw_beta[6] * cf1_simul[t-2] +
            gw_beta[7] * cf1_simul[t-3] +
            gw_beta[8] * cf1_simul[t-4] +
            # gw_beta[9] * magpty[t-1] +  # magpty is zero in IRF
            gw_beta[10] * vu_shock_series[t-1] +
            gw_beta[11] * vu_shock_series[t-2] +
            gw_beta[12] * vu_shock_series[t-3] +
            gw_beta[13] * vu_shock_series[t-4] +
            gw_beta[14] * diffcpicf_simul[t-1] +
            gw_beta[15] * diffcpicf_simul[t-2] +
            gw_beta[16] * diffcpicf_simul[t-3] +
            gw_beta[17] * diffcpicf_simul[t-4] +
            gw_beta[18] * gcu_shock_series[t-1] +  # capacity utilization
            gw_beta[19] * gcu_shock_series[t-2] +
            gw_beta[20] * gcu_shock_series[t-3] +
            gw_beta[21] * gcu_shock_series[t-4]
            # dummies are zero in IRF
        )

        # =====================================================================
        # STEP 2: UPDATE WAGE LEVEL AND COMPUTE EXCESS DEMAND (endogenous)
        # In the IRF, we track deviations from steady state:
        # - log_w_simul is the deviation of log(W) from its steady state path
        # - In steady state, gw_simul = 0, so log_w_simul stays at 0
        # - When shocked, gw_simul > 0, and log_w_simul accumulates
        # - NGDPPOT and TCU are at steady state (0 deviation), so:
        #   raw_ed_simul = log_w_simul (deviation of wages from potential)
        # =====================================================================
        log_w_simul[t] = log_w_simul[t-1] + gw_simul[t] / 400.0

        # Raw excess demand = log(W) - log(NGDPPOT) - log(TCU/100)
        # In deviations from steady state: NGDPPOT and TCU are at 0, so:
        raw_ed_simul[t] = log_w_simul[t]

        # Compute rolling trend (40-quarter rolling mean)
        lookback = min(40, t + 1)
        rolling_trend = np.mean(raw_ed_simul[t-lookback+1:t+1])

        # Detrended excess demand
        ed_simul[t] = raw_ed_simul[t] - rolling_trend

        # =====================================================================
        # STEP 3: SHORTAGE EQUATION (uses ENDOGENOUS excess demand)
        # Coefficients: [const, L1-L4 shortage, excess_demand L0-L4, gscpi L0-L4]
        # NOTE: Exclude constant for IRF (measuring deviations from steady state)
        # =====================================================================
        if add_shortage_shock and t == 4:
            # Direct shock to shortage (bypasses the shortage equation)
            shortage_simul[t] = shock_val_shortage
        else:
            shortage_simul[t] = (
                # shortage_beta[0] +  # constant EXCLUDED for IRF
                shortage_beta[1] * shortage_simul[t-1] +
                shortage_beta[2] * shortage_simul[t-2] +
                shortage_beta[3] * shortage_simul[t-3] +
                shortage_beta[4] * shortage_simul[t-4] +
                shortage_beta[5] * ed_simul[t] +  # ENDOGENOUS excess demand
                shortage_beta[6] * ed_simul[t-1] +
                shortage_beta[7] * ed_simul[t-2] +
                shortage_beta[8] * ed_simul[t-3] +
                shortage_beta[9] * ed_simul[t-4] +
                shortage_beta[10] * gscpi_shock_series[t] +
                shortage_beta[11] * gscpi_shock_series[t-1] +
                shortage_beta[12] * gscpi_shock_series[t-2] +
                shortage_beta[13] * gscpi_shock_series[t-3] +
                shortage_beta[14] * gscpi_shock_series[t-4]
            )

        # =====================================================================
        # STEP 4: PRICE EQUATION
        # Coefficients: [const, magpty, L1-L4 gcpi, gw L0-L4, grpe L0-L4,
        #                grpf L0-L4, shortage L0-L4]
        # =====================================================================
        gcpi_simul[t] = (
            # gcpi_beta[0] +  # constant (ignore for IRF from zero)
            # gcpi_beta[1] * magpty[t] +  # magpty is zero
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
            gcpi_beta[21] * shortage_simul[t] +  # Endogenous shortage
            gcpi_beta[22] * shortage_simul[t-1] +
            gcpi_beta[23] * shortage_simul[t-2] +
            gcpi_beta[24] * shortage_simul[t-3] +
            gcpi_beta[25] * shortage_simul[t-4]
        )

        # =====================================================================
        # STEP 5: CATCH-UP TERM
        # =====================================================================
        diffcpicf_simul[t] = 0.25 * (gcpi_simul[t] + gcpi_simul[t-1] +
                                     gcpi_simul[t-2] + gcpi_simul[t-3]) - cf1_simul[t-4]

        # =====================================================================
        # STEP 6: EXPECTATIONS
        # =====================================================================
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

    # Create results DataFrame
    results = pd.DataFrame({
        'period': np.arange(1, timesteps + 1),
        'gw_simul': gw_simul,
        'gcpi_simul': gcpi_simul,
        'shortage_simul': shortage_simul,
        'ed_simul': ed_simul,  # Endogenous excess demand
        'log_w_simul': log_w_simul,  # Log wage level deviation
        'cf1_simul': cf1_simul,
        'cf10_simul': cf10_simul,
        'diffcpicf_simul': diffcpicf_simul,
        'grpe_shock_series': grpe_shock_series,
        'grpf_shock_series': grpf_shock_series,
        'vu_shock_series': vu_shock_series,
        'gscpi_shock_series': gscpi_shock_series,
        'gcu_shock_series': gcu_shock_series
    })

    return results


# %%
print("\n" + "="*80)
print("RUNNING IRF SIMULATIONS")
print("="*80)

# Original BB shocks
print("\n--- Original BB Shocks ---")

print("\nEnergy prices shock:")
results_energy = irfs_new_model(
    data, table_q4_data,
    shocks={'grpe': True},
    rho={'grpe': 0.0},
    coef_path=coef_path
)

print("\nFood prices shock:")
results_food = irfs_new_model(
    data, table_q4_data,
    shocks={'grpf': True},
    rho={'grpf': 0.0},
    coef_path=coef_path
)

print("\nV/U shock (persistent):")
results_vu = irfs_new_model(
    data, table_q4_data,
    shocks={'vu': True},
    rho={'vu': 1.0},  # Permanent shock
    coef_path=coef_path
)

print("\nShortage shock (direct):")
results_shortage = irfs_new_model(
    data, table_q4_data,
    shocks={'shortage': True},
    rho={'shortage': 0.0},
    coef_path=coef_path
)

# NEW model-specific shocks
print("\n--- New Model Shocks ---")

print("\nGSCPI shock:")
results_gscpi = irfs_new_model(
    data, table_q4_data,
    shocks={'gscpi': True},
    rho={'gscpi': 0.0},
    coef_path=coef_path
)

print("\nCapacity utilization shock:")
results_gcu = irfs_new_model(
    data, table_q4_data,
    shocks={'gcu': True},
    rho={'gcu': 0.0},
    coef_path=coef_path
)

# Persistent versions
print("\n--- Persistent Shocks ---")

print("\nGSCPI shock (persistent):")
results_gscpi_persistent = irfs_new_model(
    data, table_q4_data,
    shocks={'gscpi': True},
    rho={'gscpi': 0.9},
    coef_path=coef_path
)

print("\nCapacity utilization shock (persistent):")
results_gcu_persistent = irfs_new_model(
    data, table_q4_data,
    shocks={'gcu': True},
    rho={'gcu': 0.9},
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
results_gscpi.to_excel(output_dir / 'results_gscpi.xlsx', index=False)
results_gcu.to_excel(output_dir / 'results_gcu.xlsx', index=False)
results_gscpi_persistent.to_excel(output_dir / 'results_gscpi_persistent.xlsx', index=False)
results_gcu_persistent.to_excel(output_dir / 'results_gcu_persistent.xlsx', index=False)

print(f"\nResults saved to: {output_dir}")
print("  Original BB shocks:")
print("    - results_energy.xlsx")
print("    - results_food.xlsx")
print("    - results_vu.xlsx")
print("    - results_shortage.xlsx")
print("  New model shocks:")
print("    - results_gscpi.xlsx [NEW]")
print("    - results_gcu.xlsx [NEW]")
print("  Persistent shocks:")
print("    - results_gscpi_persistent.xlsx [NEW]")
print("    - results_gcu_persistent.xlsx [NEW]")


# %%
# Print summary of IRF responses
print("\n" + "="*80)
print("IRF SUMMARY")
print("="*80)

print(f"\nOriginal BB Shocks (Peak inflation response):")
print(f"  Energy shock:     Peak gcpi = {results_energy['gcpi_simul'].max():.4f} at period {results_energy['gcpi_simul'].argmax() + 1}")
print(f"  Food shock:       Peak gcpi = {results_food['gcpi_simul'].max():.4f} at period {results_food['gcpi_simul'].argmax() + 1}")
print(f"  V/U shock:        Peak gcpi = {results_vu['gcpi_simul'].max():.4f} at period {results_vu['gcpi_simul'].argmax() + 1}")
print(f"  Shortage shock:   Peak gcpi = {results_shortage['gcpi_simul'].max():.4f} at period {results_shortage['gcpi_simul'].argmax() + 1}")

print(f"\nNew Model Shocks (one-time):")
print(f"  GSCPI:            Peak gcpi = {results_gscpi['gcpi_simul'].max():.4f} at period {results_gscpi['gcpi_simul'].argmax() + 1}")
print(f"  Capacity util:    Peak gcpi = {results_gcu['gcpi_simul'].max():.4f} at period {results_gcu['gcpi_simul'].argmax() + 1}")

print(f"\nNew Model Shocks (persistent, rho=0.9):")
print(f"  GSCPI:            Peak gcpi = {results_gscpi_persistent['gcpi_simul'].max():.4f} at period {results_gscpi_persistent['gcpi_simul'].argmax() + 1}")
print(f"  Capacity util:    Peak gcpi = {results_gcu_persistent['gcpi_simul'].max():.4f} at period {results_gcu_persistent['gcpi_simul'].argmax() + 1}")

# Show the V/U → ED → Shortage feedback
print(f"\nV/U Shock - Endogenous Feedback Loop:")
print(f"  Peak wage growth (gw):     {results_vu['gw_simul'].max():.4f} at period {results_vu['gw_simul'].argmax() + 1}")
print(f"  Peak log wage level:       {results_vu['log_w_simul'].max():.4f} at period {results_vu['log_w_simul'].argmax() + 1}")
print(f"  Peak excess demand (ed):   {results_vu['ed_simul'].max():.4f} at period {results_vu['ed_simul'].argmax() + 1}")
print(f"  Peak shortage:             {results_vu['shortage_simul'].max():.4f} at period {results_vu['shortage_simul'].argmax() + 1}")
print(f"  Peak inflation (gcpi):     {results_vu['gcpi_simul'].max():.4f} at period {results_vu['gcpi_simul'].argmax() + 1}")

print(f"\nCapacity Utilization Shock - Endogenous Feedback Loop:")
print(f"  Peak wage growth (gw):     {results_gcu['gw_simul'].max():.4f} at period {results_gcu['gw_simul'].argmax() + 1}")
print(f"  Peak log wage level:       {results_gcu['log_w_simul'].max():.4f} at period {results_gcu['log_w_simul'].argmax() + 1}")
print(f"  Peak excess demand (ed):   {results_gcu['ed_simul'].max():.4f} at period {results_gcu['ed_simul'].argmax() + 1}")
print(f"  Peak shortage:             {results_gcu['shortage_simul'].max():.4f} at period {results_gcu['shortage_simul'].argmax() + 1}")
print(f"  Peak inflation (gcpi):     {results_gcu['gcpi_simul'].max():.4f} at period {results_gcu['gcpi_simul'].argmax() + 1}")

print("\n" + "="*80)
print("IRF SIMULATIONS COMPLETE!")
print("="*80)
print("\n")

# %%
