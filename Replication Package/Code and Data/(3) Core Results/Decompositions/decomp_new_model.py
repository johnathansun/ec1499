# %%
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modified Bernanke-Blanchard Model - Decomposition
Liang & Sun (2025)

This script generates decomposition analysis for the modified model.

Key modifications from original:
1. Wage equation includes detrended capacity utilization level (cu)
2. Shortage is ENDOGENOUS: shortage = f(lagged_shortage, excess_demand, GSCPI)
3. New decomposition channels:
   - Excess demand contribution to shortages (and thus inflation)
   - GSCPI contribution to shortages (and thus inflation)
   - Capacity utilization contribution to wages (and thus inflation)

Shocks analyzed:
- Energy prices (grpe)
- Food prices (grpf)
- V/U ratio
- Shortages (total) - now endogenous
- Excess demand component of shortages [NEW]
- GSCPI component of shortages [NEW]
- Capacity utilization (cu) [NEW]
- Productivity (magpty)
- Q2 2020 dummy
- Q3 2020 dummy
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
output_dir = Path("/Users/johnathansun/Documents/ec1499/Replication Package/Code and Data/(3) Core Results/Decompositions/Output Data Python (New Model)")
output_dir.mkdir(parents=True, exist_ok=True)

#*********************************************************************************

print("="*80)
print("MODIFIED MODEL DECOMPOSITION")
print("Liang & Sun (2025)")
print("="*80)

print("\nLoading data and coefficients...")

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


def dynamic_simul_new_model(data, coef_path,
                             remove_grpe=False, remove_grpf=False,
                             remove_vu=False, remove_shortage=False,
                             remove_excess_demand=False, remove_gscpi=False,
                             remove_cu=False, remove_magpty=False,
                             remove_dummy2020_q2=False, remove_dummy2020_q3=False):
    """
    Run dynamic simulation for the modified model with specified shocks removed.

    KEY CHANGE: Excess demand is now ENDOGENOUS in counterfactual simulations.
    When wages change (due to removing V/U, CU, etc.), the wage level changes,
    which changes excess demand, which changes shortages.

    Causal chain: shock → wages → wage level → excess demand → shortage → inflation

    NEW parameters:
    - remove_excess_demand: Hold excess demand at steady state (breaks the feedback)
    - remove_gscpi: Remove GSCPI contribution to shortage equation
    - remove_cu: Remove capacity utilization from wage equation

    The simulation order is: Wages → Excess Demand → Shortage → Prices
    """

    # Load coefficients
    coeffs = load_coefficients(coef_path)
    gw_beta = coeffs['gw']
    shortage_beta = coeffs['shortage']
    gcpi_beta = coeffs['gcpi']
    cf1_beta = coeffs['cf1']
    cf10_beta = coeffs['cf10']

    # Track which shocks are removed
    shocks_removed = []
    if remove_grpe: shocks_removed.append("grpe")
    if remove_grpf: shocks_removed.append("grpf")
    if remove_vu: shocks_removed.append("vu")
    if remove_shortage: shocks_removed.append("shortage")
    if remove_excess_demand: shocks_removed.append("excess_demand")
    if remove_gscpi: shocks_removed.append("gscpi")
    if remove_cu: shocks_removed.append("cu")
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
    shortage = data['shortage'].values.copy()
    diffcpicf = data['diffcpicf'].values.copy()

    # Exogenous variables (historical)
    grpe = data['grpe'].values.copy()
    grpf = data['grpf'].values.copy()
    vu = data['vu'].values.copy()
    magpty = data['magpty'].values.copy()

    # New model variables
    cu = data['cu'].values.copy() if 'cu' in data.columns else np.zeros(timesteps)
    excess_demand = data['excess_demand'].values.copy() if 'excess_demand' in data.columns else np.zeros(timesteps)
    gscpi = data['gscpi'].values.copy() if 'gscpi' in data.columns else np.zeros(timesteps)

    # Handle NaN in new variables
    cu = np.nan_to_num(cu, nan=0.0)
    excess_demand = np.nan_to_num(excess_demand, nan=excess_demand[~np.isnan(excess_demand)].mean() if np.any(~np.isnan(excess_demand)) else 0.0)
    gscpi = np.nan_to_num(gscpi, nan=0.0)

    # =========================================================================
    # LEVEL TRACKING FOR ENDOGENOUS EXCESS DEMAND
    # =========================================================================
    # excess_demand = log(W) - log(NGDPPOT) - log(TCU/100) - trend
    # We need to track wage and potential GDP levels to compute ED endogenously

    # Get historical levels from data
    log_w = data['log_w'].values.copy() if 'log_w' in data.columns else None
    log_ngdppot = data['log_ngdppot'].values.copy() if 'log_ngdppot' in data.columns else None
    tcu = data['tcu'].values.copy() if 'tcu' in data.columns else np.full(timesteps, 75.0)

    # Steady-state parameters for excess demand calculation
    log_tcu_star = np.log(0.75)  # Steady state TCU = 75%
    g_ngdppot = 4.0  # Nominal potential GDP growth rate (annualized)

    # If we don't have log_w, reconstruct from wage growth
    if log_w is None:
        print("  Warning: log_w not in data, reconstructing from wage growth")
        log_w = np.zeros(timesteps)
        log_w[0] = np.log(150)  # Approximate ECI index starting value
        for i in range(1, timesteps):
            log_w[i] = log_w[i-1] + gw[i] / 400

    # If we don't have log_ngdppot, reconstruct from growth rate
    if log_ngdppot is None:
        print("  Warning: log_ngdppot not in data, reconstructing from growth rate")
        log_ngdppot = np.zeros(timesteps)
        log_ngdppot[0] = np.log(25000)  # Approximate NGDPPOT starting value
        for i in range(1, timesteps):
            log_ngdppot[i] = log_ngdppot[i-1] + g_ngdppot / 400

    # Compute log_tcu from tcu
    log_tcu = np.log(tcu / 100)

    # Compute raw excess demand for historical data (for rolling trend)
    raw_excess_demand_hist = log_w - log_ngdppot - log_tcu

    # Define dummy variables
    dummy_q2 = np.zeros(timesteps)
    dummy_q3 = np.zeros(timesteps)
    for i, p in enumerate(period):
        p_dt = pd.Timestamp(p)
        if p_dt.year == 2020 and p_dt.quarter == 2:
            dummy_q2[i] = 1.0
        if p_dt.year == 2020 and p_dt.quarter == 3:
            dummy_q3[i] = 1.0

    # Initialize simulation arrays (first 4 values from historical data)
    gw_simul = np.zeros(timesteps)
    gcpi_simul = np.zeros(timesteps)
    cf1_simul = np.zeros(timesteps)
    cf10_simul = np.zeros(timesteps)
    shortage_simul = np.zeros(timesteps)
    diffcpicf_simul = np.zeros(timesteps)
    excess_demand_simul = np.zeros(timesteps)

    # Level tracking arrays for counterfactual
    log_w_simul = np.zeros(timesteps)
    log_ngdppot_simul = np.zeros(timesteps)
    raw_excess_demand_simul = np.zeros(timesteps)

    # Initialize from historical data
    gw_simul[:4] = gw[:4]
    gcpi_simul[:4] = gcpi[:4]
    cf1_simul[:4] = cf1[:4]
    cf10_simul[:4] = cf10[:4]
    shortage_simul[:4] = shortage[:4]
    diffcpicf_simul[:4] = diffcpicf[:4]
    excess_demand_simul[:4] = excess_demand[:4]

    # Initialize levels from historical data
    log_w_simul[:4] = log_w[:4]
    log_ngdppot_simul[:4] = log_ngdppot[:4]
    raw_excess_demand_simul[:4] = raw_excess_demand_hist[:4]

    # Initialize shock series for simulation
    grpe_simul = grpe.copy()
    grpf_simul = grpf.copy()
    vu_simul = vu.copy()
    magpty_simul = magpty.copy()
    cu_simul = cu.copy()
    gscpi_simul = gscpi.copy()
    dummy_q2_simul = dummy_q2.copy()
    dummy_q3_simul = dummy_q3.copy()

    # Get steady-state values for counterfactuals
    vu_steady = data['vu'].iloc[4] if len(data) > 4 else 1.2
    shortage_steady = 5.0
    excess_demand_steady = 0.0  # On trend
    cu_steady = 0.0  # Detrended capacity utilization = 0 means on trend

    # Run counterfactual simulation
    for t in range(4, timesteps):

        # Set exogenous variables based on removal flags
        if remove_grpe:
            grpe_simul[t] = 0

        if remove_grpf:
            grpf_simul[t] = 0

        if remove_vu:
            vu_simul[t] = vu_steady

        if remove_cu:
            cu_simul[t] = cu_steady

        if remove_magpty:
            magpty_simul[t] = 2.0  # Long-run average

        if remove_dummy2020_q2:
            dummy_q2_simul[t] = 0.0

        if remove_dummy2020_q3:
            dummy_q3_simul[t] = 0.0

        if remove_gscpi:
            gscpi_simul[t] = 0.0

        # =====================================================================
        # STEP 1: WAGE EQUATION (computed first since ED depends on wages)
        # =====================================================================
        gw_simul[t] = (
            gw_beta[0] +  # constant
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
            gw_beta[22] * dummy_q2_simul[t] +
            gw_beta[23] * dummy_q3_simul[t]
        )

        # =====================================================================
        # STEP 2: UPDATE LEVELS AND COMPUTE EXCESS DEMAND (endogenous)
        # =====================================================================
        # Update wage level from simulated wage growth
        log_w_simul[t] = log_w_simul[t-1] + gw_simul[t] / 400

        # Update potential GDP level (exogenous growth)
        log_ngdppot_simul[t] = log_ngdppot_simul[t-1] + g_ngdppot / 400

        # Compute raw excess demand from simulated levels
        # Use historical TCU (exogenous) but simulated wage level
        raw_excess_demand_simul[t] = log_w_simul[t] - log_ngdppot_simul[t] - log_tcu[t]

        # Compute rolling trend (40-quarter rolling mean)
        lookback = min(40, t + 1)
        rolling_trend = np.mean(raw_excess_demand_simul[t-lookback+1:t+1])

        # Detrend excess demand
        if remove_excess_demand:
            # If removing excess demand effect, hold at steady state
            excess_demand_simul[t] = excess_demand_steady
        else:
            # Endogenous excess demand from simulated wage level
            excess_demand_simul[t] = raw_excess_demand_simul[t] - rolling_trend

        # =====================================================================
        # STEP 3: SHORTAGE EQUATION (uses endogenous excess demand)
        # =====================================================================
        if remove_shortage:
            shortage_simul[t] = shortage_steady
        else:
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

        # =====================================================================
        # STEP 4: PRICE EQUATION (uses endogenous shortage)
        # =====================================================================
        gcpi_simul[t] = (
            gcpi_beta[0] +  # constant
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
            gcpi_beta[25] * shortage_simul[t-4]
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

    # =========================================================================
    # RUN BASELINE SIMULATION (all shocks included, ED also endogenous)
    # =========================================================================
    gw_baseline = np.zeros(timesteps)
    gcpi_baseline = np.zeros(timesteps)
    cf1_baseline = np.zeros(timesteps)
    cf10_baseline = np.zeros(timesteps)
    shortage_baseline = np.zeros(timesteps)
    diffcpicf_baseline = np.zeros(timesteps)
    excess_demand_baseline = np.zeros(timesteps)

    # Level tracking for baseline
    log_w_baseline = np.zeros(timesteps)
    log_ngdppot_baseline = np.zeros(timesteps)
    raw_excess_demand_baseline = np.zeros(timesteps)

    # Initialize from historical data
    gw_baseline[:4] = gw[:4]
    gcpi_baseline[:4] = gcpi[:4]
    cf1_baseline[:4] = cf1[:4]
    cf10_baseline[:4] = cf10[:4]
    shortage_baseline[:4] = shortage[:4]
    diffcpicf_baseline[:4] = diffcpicf[:4]
    excess_demand_baseline[:4] = excess_demand[:4]

    log_w_baseline[:4] = log_w[:4]
    log_ngdppot_baseline[:4] = log_ngdppot[:4]
    raw_excess_demand_baseline[:4] = raw_excess_demand_hist[:4]

    for t in range(4, timesteps):
        # STEP 1: Wage equation (baseline)
        gw_baseline[t] = (
            gw_beta[0] +
            gw_beta[1] * gw_baseline[t-1] +
            gw_beta[2] * gw_baseline[t-2] +
            gw_beta[3] * gw_baseline[t-3] +
            gw_beta[4] * gw_baseline[t-4] +
            gw_beta[5] * cf1_baseline[t-1] +
            gw_beta[6] * cf1_baseline[t-2] +
            gw_beta[7] * cf1_baseline[t-3] +
            gw_beta[8] * cf1_baseline[t-4] +
            gw_beta[9] * magpty[t-1] +
            gw_beta[10] * vu[t-1] +
            gw_beta[11] * vu[t-2] +
            gw_beta[12] * vu[t-3] +
            gw_beta[13] * vu[t-4] +
            gw_beta[14] * diffcpicf_baseline[t-1] +
            gw_beta[15] * diffcpicf_baseline[t-2] +
            gw_beta[16] * diffcpicf_baseline[t-3] +
            gw_beta[17] * diffcpicf_baseline[t-4] +
            gw_beta[18] * cu[t-1] +
            gw_beta[19] * cu[t-2] +
            gw_beta[20] * cu[t-3] +
            gw_beta[21] * cu[t-4] +
            gw_beta[22] * dummy_q2[t] +
            gw_beta[23] * dummy_q3[t]
        )

        # STEP 2: Update levels and compute excess demand (endogenous)
        log_w_baseline[t] = log_w_baseline[t-1] + gw_baseline[t] / 400
        log_ngdppot_baseline[t] = log_ngdppot_baseline[t-1] + g_ngdppot / 400
        raw_excess_demand_baseline[t] = log_w_baseline[t] - log_ngdppot_baseline[t] - log_tcu[t]

        lookback = min(40, t + 1)
        rolling_trend_baseline = np.mean(raw_excess_demand_baseline[t-lookback+1:t+1])
        excess_demand_baseline[t] = raw_excess_demand_baseline[t] - rolling_trend_baseline

        # STEP 3: Shortage equation (baseline)
        shortage_baseline[t] = (
            shortage_beta[0] +
            shortage_beta[1] * shortage_baseline[t-1] +
            shortage_beta[2] * shortage_baseline[t-2] +
            shortage_beta[3] * shortage_baseline[t-3] +
            shortage_beta[4] * shortage_baseline[t-4] +
            shortage_beta[5] * excess_demand_baseline[t] +
            shortage_beta[6] * excess_demand_baseline[t-1] +
            shortage_beta[7] * excess_demand_baseline[t-2] +
            shortage_beta[8] * excess_demand_baseline[t-3] +
            shortage_beta[9] * excess_demand_baseline[t-4] +
            shortage_beta[10] * gscpi[t] +
            shortage_beta[11] * gscpi[t-1] +
            shortage_beta[12] * gscpi[t-2] +
            shortage_beta[13] * gscpi[t-3] +
            shortage_beta[14] * gscpi[t-4]
        )

        # STEP 4: Price equation (baseline)
        gcpi_baseline[t] = (
            gcpi_beta[0] +
            gcpi_beta[1] * magpty[t] +
            gcpi_beta[2] * gcpi_baseline[t-1] +
            gcpi_beta[3] * gcpi_baseline[t-2] +
            gcpi_beta[4] * gcpi_baseline[t-3] +
            gcpi_beta[5] * gcpi_baseline[t-4] +
            gcpi_beta[6] * gw_baseline[t] +
            gcpi_beta[7] * gw_baseline[t-1] +
            gcpi_beta[8] * gw_baseline[t-2] +
            gcpi_beta[9] * gw_baseline[t-3] +
            gcpi_beta[10] * gw_baseline[t-4] +
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
            gcpi_beta[21] * shortage_baseline[t] +
            gcpi_beta[22] * shortage_baseline[t-1] +
            gcpi_beta[23] * shortage_baseline[t-2] +
            gcpi_beta[24] * shortage_baseline[t-3] +
            gcpi_beta[25] * shortage_baseline[t-4]
        )

        diffcpicf_baseline[t] = 0.25 * (gcpi_baseline[t] + gcpi_baseline[t-1] +
                                         gcpi_baseline[t-2] + gcpi_baseline[t-3]) - cf1_baseline[t-4]

        cf10_baseline[t] = (
            cf10_beta[0] * cf10_baseline[t-1] +
            cf10_beta[1] * cf10_baseline[t-2] +
            cf10_beta[2] * cf10_baseline[t-3] +
            cf10_beta[3] * cf10_baseline[t-4] +
            cf10_beta[4] * gcpi_baseline[t] +
            cf10_beta[5] * gcpi_baseline[t-1] +
            cf10_beta[6] * gcpi_baseline[t-2] +
            cf10_beta[7] * gcpi_baseline[t-3] +
            cf10_beta[8] * gcpi_baseline[t-4]
        )

        cf1_baseline[t] = (
            cf1_beta[0] * cf1_baseline[t-1] +
            cf1_beta[1] * cf1_baseline[t-2] +
            cf1_beta[2] * cf1_baseline[t-3] +
            cf1_beta[3] * cf1_baseline[t-4] +
            cf1_beta[4] * cf10_baseline[t] +
            cf1_beta[5] * cf10_baseline[t-1] +
            cf1_beta[6] * cf10_baseline[t-2] +
            cf1_beta[7] * cf10_baseline[t-3] +
            cf1_beta[8] * cf10_baseline[t-4] +
            cf1_beta[9] * gcpi_baseline[t] +
            cf1_beta[10] * gcpi_baseline[t-1] +
            cf1_beta[11] * gcpi_baseline[t-2] +
            cf1_beta[12] * gcpi_baseline[t-3] +
            cf1_beta[13] * gcpi_baseline[t-4]
        )

    # Create output DataFrame
    out_data = pd.DataFrame({
        'period': period,
        'gw': gw,
        'gw_simul': gw_simul,
        'gw_baseline': gw_baseline,
        'gcpi': gcpi,
        'gcpi_simul': gcpi_simul,
        'gcpi_baseline': gcpi_baseline,
        'shortage': shortage,
        'shortage_simul': shortage_simul,
        'shortage_baseline': shortage_baseline,
        'cf1': cf1,
        'cf1_simul': cf1_simul,
        'cf1_baseline': cf1_baseline,
        'cf10': cf10,
        'cf10_simul': cf10_simul,
        'cf10_baseline': cf10_baseline,
        'grpe': grpe,
        'grpf': grpf,
        'vu': vu,
        'cu': cu,
        'excess_demand': excess_demand,
        'excess_demand_simul': excess_demand_simul,  # NEW: simulated ED
        'excess_demand_baseline': excess_demand_baseline,  # NEW: baseline ED
        'gscpi': gscpi,
        'magpty': magpty
    })

    # Calculate contributions if only one shock is removed
    if len(shocks_removed) == 1:
        shock_name = shocks_removed[0]
        out_data[f'{shock_name}_contr_gw'] = gw_baseline - gw_simul
        out_data[f'{shock_name}_contr_gcpi'] = gcpi_baseline - gcpi_simul
        out_data[f'{shock_name}_contr_shortage'] = shortage_baseline - shortage_simul
        out_data[f'{shock_name}_contr_cf1'] = cf1_baseline - cf1_simul
        out_data[f'{shock_name}_contr_cf10'] = cf10_baseline - cf10_simul
        # NEW: contribution to excess demand (for shocks that affect wages)
        out_data[f'{shock_name}_contr_excess_demand'] = excess_demand_baseline - excess_demand_simul

    return out_data


# %%
print("\n" + "="*80)
print("RUNNING DECOMPOSITION SIMULATIONS")
print("="*80)

# Run all decomposition scenarios

# Original BB shocks
print("\n--- Original BB Shocks ---")

print("\nBaseline (all shocks active):")
baseline = dynamic_simul_new_model(data, coef_path)

print("\nRemove energy prices:")
remove_grpe = dynamic_simul_new_model(data, coef_path, remove_grpe=True)

print("\nRemove food prices:")
remove_grpf = dynamic_simul_new_model(data, coef_path, remove_grpf=True)

print("\nRemove V/U:")
remove_vu = dynamic_simul_new_model(data, coef_path, remove_vu=True)

print("\nRemove shortage (total):")
remove_shortage = dynamic_simul_new_model(data, coef_path, remove_shortage=True)

print("\nRemove productivity:")
remove_magpty = dynamic_simul_new_model(data, coef_path, remove_magpty=True)

print("\nRemove Q2 2020 dummy:")
remove_q2 = dynamic_simul_new_model(data, coef_path, remove_dummy2020_q2=True)

print("\nRemove Q3 2020 dummy:")
remove_q3 = dynamic_simul_new_model(data, coef_path, remove_dummy2020_q3=True)

# NEW shocks from modified model
print("\n--- New Model Shocks ---")

print("\nRemove excess demand (from shortage equation):")
remove_excess_demand = dynamic_simul_new_model(data, coef_path, remove_excess_demand=True)

print("\nRemove GSCPI (from shortage equation):")
remove_gscpi = dynamic_simul_new_model(data, coef_path, remove_gscpi=True)

print("\nRemove capacity utilization (from wage equation):")
remove_cu = dynamic_simul_new_model(data, coef_path, remove_cu=True)

# Combined removal
print("\nRemove all shocks:")
remove_all = dynamic_simul_new_model(data, coef_path,
                                      remove_grpe=True, remove_grpf=True, remove_vu=True,
                                      remove_shortage=True, remove_cu=True, remove_magpty=True,
                                      remove_dummy2020_q2=True, remove_dummy2020_q3=True)


# %%
print("\n" + "="*80)
print("EXPORTING RESULTS")
print("="*80)

# Export to Excel
baseline.to_excel(output_dir / 'baseline.xlsx', index=False)
remove_grpe.to_excel(output_dir / 'remove_grpe.xlsx', index=False)
remove_grpf.to_excel(output_dir / 'remove_grpf.xlsx', index=False)
remove_vu.to_excel(output_dir / 'remove_vu.xlsx', index=False)
remove_shortage.to_excel(output_dir / 'remove_shortage.xlsx', index=False)
remove_magpty.to_excel(output_dir / 'remove_magpty.xlsx', index=False)
remove_q2.to_excel(output_dir / 'remove_2020q2.xlsx', index=False)
remove_q3.to_excel(output_dir / 'remove_2020q3.xlsx', index=False)
remove_excess_demand.to_excel(output_dir / 'remove_excess_demand.xlsx', index=False)
remove_gscpi.to_excel(output_dir / 'remove_gscpi.xlsx', index=False)
remove_cu.to_excel(output_dir / 'remove_cu.xlsx', index=False)
remove_all.to_excel(output_dir / 'remove_all.xlsx', index=False)

print(f"\nResults saved to: {output_dir}")
print("  Original BB shocks:")
print("    - baseline.xlsx")
print("    - remove_grpe.xlsx")
print("    - remove_grpf.xlsx")
print("    - remove_vu.xlsx")
print("    - remove_shortage.xlsx")
print("    - remove_magpty.xlsx")
print("    - remove_2020q2.xlsx")
print("    - remove_2020q3.xlsx")
print("  New model shocks:")
print("    - remove_excess_demand.xlsx [NEW]")
print("    - remove_gscpi.xlsx [NEW]")
print("    - remove_cu.xlsx [NEW]")
print("    - remove_all.xlsx")


# %%
# =============================================================================
# EXCESS DEMAND COMPONENT ATTRIBUTION
# =============================================================================
# excess_demand = log(wages) - log(potential GDP) - log(capacity utilization)
# We can decompose how much of excess_demand change comes from each component
print("\n" + "="*80)
print("COMPUTING EXCESS DEMAND COMPONENT ATTRIBUTION")
print("="*80)

# Check if we have the component variables
has_components = all(col in data.columns for col in ['log_w', 'log_ngdppot', 'cu'])

if not has_components:
    # Try to compute from raw variables if available
    print("  Computing log components from raw data...")
    if 'tcu' in data.columns and 'log_tcu' not in data.columns:
        data['log_tcu'] = np.log(data['tcu'] / 100)
    if 'ngdppot' in data.columns and 'log_ngdppot' not in data.columns:
        data['log_ngdppot'] = np.log(data['ngdppot'])
    # Note: log_w should already be computed in regression

# Compute changes from reference period (Q4 2019 = pre-COVID steady state)
ref_idx = data[data['period'] >= '2019-10-01'].index[0] if len(data[data['period'] >= '2019-10-01']) > 0 else 4

# Get reference values
excess_demand_ref = data['excess_demand'].iloc[ref_idx] if 'excess_demand' in data.columns else 0

# Compute excess_demand deviation from reference
excess_demand_deviation = data['excess_demand'] - excess_demand_ref

# For component attribution, we need to decompose the excess_demand change
# excess_demand = log(w) - log(ngdppot) - log(cu)
# Δexcess_demand = Δlog(w) - Δlog(ngdppot) - Δlog(cu)

# excess_demand = log(w) - log(ngdppot) - log(tcu/100) - trend
# We have log_w, log_ngdppot, and can compute log_tcu from tcu
if 'log_w' in data.columns and 'log_ngdppot' in data.columns and 'tcu' in data.columns:
    # Compute log_tcu if not already present
    if 'log_tcu' not in data.columns:
        data['log_tcu'] = np.log(data['tcu'] / 100)

    log_w_ref = data['log_w'].iloc[ref_idx]
    log_ngdppot_ref = data['log_ngdppot'].iloc[ref_idx]
    log_tcu_ref = data['log_tcu'].iloc[ref_idx]

    # Contribution of wages to excess_demand (positive sign: higher wages -> higher excess demand)
    wage_contr_to_ed = data['log_w'] - log_w_ref

    # Contribution of potential GDP to excess_demand (negative sign: higher ngdppot -> lower excess demand)
    ngdppot_contr_to_ed = -(data['log_ngdppot'] - log_ngdppot_ref)

    # Contribution of capacity utilization to excess_demand (negative sign: higher cu -> lower excess demand)
    cu_contr_to_ed = -(data['log_tcu'] - log_tcu_ref)

    print(f"  Reference period: {data['period'].iloc[ref_idx]}")
    print(f"  Computed component contributions to excess_demand")
else:
    # If we don't have components, create proportional attribution based on cu and gw
    print("  Using cumulative changes for proportional attribution...")

    # Use cu (detrended capacity utilization) directly
    # Higher cu means capacity is above trend, which reduces excess demand
    cu_ref = data['cu'].iloc[ref_idx] if 'cu' in data.columns else 0
    cu_contr_to_ed = -(data['cu'] - cu_ref) if 'cu' in data.columns else pd.Series(0, index=data.index)

    # For wages, use cumulative wage growth relative to productivity
    # Positive because higher wage growth means higher excess_demand
    gw_cumsum = (data['gw'] - data['magpty']).cumsum() / 400  # Convert to log-like units

    # Normalize to match total excess_demand change
    total_change = excess_demand_deviation

    # Simple proportional attribution based on cumulative changes
    wage_contr_to_ed = gw_cumsum   # Higher wages = higher excess demand
    ngdppot_contr_to_ed = total_change - cu_contr_to_ed - wage_contr_to_ed  # Residual for potential GDP

# Now compute how these components contribute to shortages and inflation
# The key is: excess_demand → shortage → inflation

# Get the contribution of excess_demand to shortage and inflation from our simulation
ed_contr_shortage = remove_excess_demand['excess_demand_contr_shortage']
ed_contr_gcpi = remove_excess_demand['excess_demand_contr_gcpi']

# Proportional attribution of excess_demand contribution to components
# (based on how much each component contributed to excess_demand deviation)
total_ed_deviation = excess_demand_deviation.replace(0, np.nan)  # Avoid division by zero

wage_share = wage_contr_to_ed / total_ed_deviation
cu_share = cu_contr_to_ed / total_ed_deviation
ngdppot_share = ngdppot_contr_to_ed / total_ed_deviation

# Fill NaN with 0 for early periods
wage_share = wage_share.fillna(0)
cu_share = cu_share.fillna(0)
ngdppot_share = ngdppot_share.fillna(0)

# Compute attributed contributions
wage_contr_shortage = ed_contr_shortage * wage_share
cu_contr_shortage = ed_contr_shortage * cu_share
ngdppot_contr_shortage = ed_contr_shortage * ngdppot_share

wage_contr_gcpi_via_ed = ed_contr_gcpi * wage_share
cu_contr_gcpi_via_ed = ed_contr_gcpi * cu_share
ngdppot_contr_gcpi_via_ed = ed_contr_gcpi * ngdppot_share

# Add to baseline dataframe for export
baseline['wage_contr_to_ed'] = wage_contr_to_ed
baseline['cu_contr_to_ed'] = cu_contr_to_ed
baseline['ngdppot_contr_to_ed'] = ngdppot_contr_to_ed
baseline['wage_contr_shortage'] = wage_contr_shortage
baseline['cu_contr_shortage'] = cu_contr_shortage
baseline['ngdppot_contr_shortage'] = ngdppot_contr_shortage
baseline['wage_contr_gcpi_via_ed'] = wage_contr_gcpi_via_ed
baseline['cu_contr_gcpi_via_ed'] = cu_contr_gcpi_via_ed
baseline['ngdppot_contr_gcpi_via_ed'] = ngdppot_contr_gcpi_via_ed

# Also add the cu contribution through wages (direct effect)
baseline['cu_contr_gcpi'] = remove_cu['cu_contr_gcpi']
baseline['cu_contr_gw'] = remove_cu['cu_contr_gw']

# Save updated baseline with component attributions
baseline.to_excel(output_dir / 'baseline.xlsx', index=False)

# Create separate export for component decomposition
component_decomp = pd.DataFrame({
    'period': baseline['period'],
    # Excess demand components
    'excess_demand': baseline['excess_demand'],
    'excess_demand_deviation': excess_demand_deviation,
    'wage_contr_to_ed': wage_contr_to_ed,
    'cu_contr_to_ed': cu_contr_to_ed,
    'ngdppot_contr_to_ed': ngdppot_contr_to_ed,
    # Contributions to shortage
    'ed_contr_shortage': ed_contr_shortage,
    'wage_contr_shortage': wage_contr_shortage,
    'cu_contr_shortage': cu_contr_shortage,
    'ngdppot_contr_shortage': ngdppot_contr_shortage,
    # Contributions to inflation via excess demand
    'ed_contr_gcpi': ed_contr_gcpi,
    'wage_contr_gcpi_via_ed': wage_contr_gcpi_via_ed,
    'cu_contr_gcpi_via_ed': cu_contr_gcpi_via_ed,
    'ngdppot_contr_gcpi_via_ed': ngdppot_contr_gcpi_via_ed,
    # Direct capacity util effect on inflation (via wages)
    'cu_contr_gcpi_direct': remove_cu['cu_contr_gcpi'],
    'cu_contr_gw': remove_cu['cu_contr_gw'],
    # Total capacity util effect on inflation (direct + via shortage)
    'cu_total_contr_gcpi': remove_cu['cu_contr_gcpi'] + cu_contr_gcpi_via_ed
})

component_decomp.to_excel(output_dir / 'excess_demand_components.xlsx', index=False)
print(f"\n  Saved excess_demand_components.xlsx")

# %%
# Print summary
print("\n" + "="*80)
print("DECOMPOSITION SUMMARY")
print("="*80)

# Filter to 2020+ for summary
summary_mask = baseline['period'] >= '2020-01-01'

print(f"\nPeak contributions to INFLATION (gcpi) from 2020 onwards:")
print("-"*60)
print(f"  Energy prices:         {remove_grpe.loc[summary_mask, 'grpe_contr_gcpi'].max():.2f}")
print(f"  Food prices:           {remove_grpf.loc[summary_mask, 'grpf_contr_gcpi'].max():.2f}")
print(f"  V/U ratio:             {remove_vu.loc[summary_mask, 'vu_contr_gcpi'].max():.2f}")
print(f"  Shortage (total):      {remove_shortage.loc[summary_mask, 'shortage_contr_gcpi'].max():.2f}")
print(f"  Productivity:          {remove_magpty.loc[summary_mask, 'magpty_contr_gcpi'].max():.2f}")
print(f"  Capacity utilization:  {remove_cu.loc[summary_mask, 'cu_contr_gcpi'].max():.2f} [NEW]")

print(f"\nPeak contributions to SHORTAGE from 2020 onwards:")
print("-"*60)
print(f"  Excess demand:         {remove_excess_demand.loc[summary_mask, 'excess_demand_contr_shortage'].max():.2f} [NEW]")
print(f"  GSCPI:                 {remove_gscpi.loc[summary_mask, 'gscpi_contr_shortage'].max():.2f} [NEW]")

print(f"\nPeak contributions to WAGES (gw) from 2020 onwards:")
print("-"*60)
print(f"  V/U ratio:             {remove_vu.loc[summary_mask, 'vu_contr_gw'].max():.2f}")
print(f"  Capacity utilization:  {remove_cu.loc[summary_mask, 'cu_contr_gw'].max():.2f} [NEW]")

print("\n" + "="*80)
print("KEY INSIGHT: SHORTAGE DECOMPOSITION")
print("="*80)
print("\nThe shortage term in the price equation can now be decomposed into:")
print("  1. Demand-driven shortages (excess demand = wages/capacity)")
print("  2. Supply-chain-driven shortages (GSCPI)")
print("\nThis allows us to answer: How much of the 'shortage-driven' inflation")
print("was actually caused by demand outpacing supply capacity?")

print("\n" + "="*80)
print("DECOMPOSITION COMPLETE!")
print("="*80)
print("\n")

# %%
