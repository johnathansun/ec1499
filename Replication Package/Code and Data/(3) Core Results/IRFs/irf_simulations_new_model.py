# %%
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modified Bernanke-Blanchard Model - Impulse Response Functions
Liang & Sun (2025)

This script generates IRFs for the modified model with support for multiple
model specifications.

Configuration flags at the top control:
- USE_PRE_COVID_SAMPLE: Whether to use pre-COVID sample estimates
- USE_LOG_CU_WAGES: Use log(CU) vs level CU in wage equation
- USE_CONTEMP_CU: Include contemporaneous CU (L0) in wage equation
- USE_DETRENDED_EXCESS_DEMAND: Detrend excess demand in shortage equation

IMPORTANT: Shock magnitudes differ significantly between specifications:
- Level CU: std dev ~ 0.04 (i.e., 4 percentage points deviation from trend)
- Log CU: std dev ~ 0.005-0.01 (log units)
This script automatically calculates shock magnitudes from the data.

Transmission channels:
- Energy/Food: grpe/grpf -> gcpi (direct)
- V/U: vu -> gw -> W -> ed -> shortage -> gcpi (full feedback loop)
- GSCPI: gscpi -> shortage -> gcpi
- Capacity utilization: TWO effects on excess demand:
  1. cu -> gw -> W -> +ed (wage channel, increases ED)
  2. cu -> TCU -> -ed (direct capacity channel, decreases ED)
  Net effect depends on coefficient magnitudes. The direct TCU effect
  partially offsets the wage-driven excess demand increase.
- Potential GDP (NGDPPOT): ngdppot -> -ed -> shortage -> gcpi
  Higher potential GDP increases effective capacity, reducing excess demand
  and thus shortages, leading to lower inflation.
- Shortage: shortage -> gcpi (direct, bypasses shortage equation)
"""

import pandas as pd
import numpy as np
from pathlib import Path

# %%
#****************************CONFIGURATION**************************************
# These flags MUST match the specification used in the regression!

USE_PRE_COVID_SAMPLE = False       # Use pre-COVID sample estimates
USE_LOG_CU_WAGES = False          # True = log(CU), False = level CU
USE_CONTEMP_CU = False             # True = CU lags 0-4, False = CU lags 1-4
USE_DETRENDED_EXCESS_DEMAND = True  # Detrend excess demand in shortage eq

#*******************************************************************************
# SHOCK CONFIGURATION
# Override automatic shock calculation by setting values here
# Leave as None to calculate from data (1 standard deviation)
#*******************************************************************************

SHOCK_OVERRIDES = {
    'grpe': None,       # Energy price shock (annualized %, typical ~30-50)
    'grpf': None,       # Food price shock (annualized %, typical ~10-20)
    'vu': None,         # V/U ratio shock (ratio, typical ~0.3-0.5)
    'shortage': None,   # Shortage index shock (Google Trends units, typical ~10-20)
    'gscpi': None,      # GSCPI shock (std dev units, typical ~1-2)
    'cu': None,         # Capacity utilization shock - NOTE: differs by spec!
                        #   Level CU: typical ~0.03-0.05 (percentage points/100)
                        #   Log CU: typical ~0.005-0.01 (log units)
    'ngdppot': None,    # Potential GDP shock (log deviation, typical ~0.01-0.02)
}

#****************************PATH CONFIGURATION*********************************

BASE_DIR = Path("/Users/johnathansun/Documents/ec1499/Replication Package/Code and Data")

# Build input directory name based on configuration (must match regression output)
def get_spec_dir_name():
    """Build output directory name based on configuration flags."""
    parts = ["Output Data (New"]
    if USE_PRE_COVID_SAMPLE:
        parts.append("Pre Covid")
    if USE_LOG_CU_WAGES:
        parts.append("Log CU")
    if USE_CONTEMP_CU:
        parts.append("Contemp CU")
    if USE_DETRENDED_EXCESS_DEMAND:
        parts.append("Detrended ED")
    return " ".join(parts) + ")"

SPEC_DIR_NAME = get_spec_dir_name()
coef_path = BASE_DIR / "(2) Regressions" / SPEC_DIR_NAME / "eq_coefficients_new_model.xlsx"
data_path = BASE_DIR / "(2) Regressions" / SPEC_DIR_NAME / "eq_simulations_data_new_model.xlsx"

# Output location for IRF results
output_dir = BASE_DIR / "(3) Core Results/IRFs/Output Data Python (New Model)" / SPEC_DIR_NAME
output_dir.mkdir(parents=True, exist_ok=True)

#*******************************************************************************

def print_configuration():
    """Print current configuration for verification."""
    print("=" * 80)
    print("MODIFIED MODEL IRF SIMULATIONS")
    print("Liang & Sun (2025)")
    print("=" * 80)
    print("\nCONFIGURATION:")
    print(f"  USE_PRE_COVID_SAMPLE:        {USE_PRE_COVID_SAMPLE}")
    print(f"  USE_LOG_CU_WAGES:            {USE_LOG_CU_WAGES}")
    print(f"  USE_CONTEMP_CU:              {USE_CONTEMP_CU}")
    print(f"  USE_DETRENDED_EXCESS_DEMAND: {USE_DETRENDED_EXCESS_DEMAND}")
    print(f"\nSpecification directory: {SPEC_DIR_NAME}")
    print(f"Coefficient file: {coef_path}")
    print(f"Data file: {data_path}")
    print(f"Output directory: {output_dir}")

print_configuration()

# %%
# VERIFY PATHS EXIST
if not coef_path.exists():
    raise FileNotFoundError(
        f"Coefficient file not found: {coef_path}\n"
        f"Make sure you have run the regression with the matching configuration:\n"
        f"  USE_PRE_COVID_SAMPLE={USE_PRE_COVID_SAMPLE}\n"
        f"  USE_LOG_CU_WAGES={USE_LOG_CU_WAGES}\n"
        f"  USE_CONTEMP_CU={USE_CONTEMP_CU}\n"
        f"  USE_DETRENDED_EXCESS_DEMAND={USE_DETRENDED_EXCESS_DEMAND}"
    )

if not data_path.exists():
    raise FileNotFoundError(f"Data file not found: {data_path}")

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
    """
    Load coefficients from Excel file (new model with 5 equations).

    Returns dict with coefficient arrays and metadata about structure.
    """
    gw_beta = pd.read_excel(coef_path, sheet_name='gw')
    shortage_beta = pd.read_excel(coef_path, sheet_name='shortage')
    gcpi_beta = pd.read_excel(coef_path, sheet_name='gcpi')
    cf1_beta = pd.read_excel(coef_path, sheet_name='cf1')
    cf10_beta = pd.read_excel(coef_path, sheet_name='cf10')

    # Verify coefficient count matches expected structure
    expected_gw_count = 23 if USE_CONTEMP_CU else 22  # 22 base + 1 for contemporaneous CU
    if USE_PRE_COVID_SAMPLE:
        expected_gw_count -= 2  # No COVID dummies in pre-COVID sample

    actual_gw_count = len(gw_beta)
    print(f"\n  Wage equation coefficients: {actual_gw_count} (expected ~{expected_gw_count})")

    return {
        'gw': gw_beta['beta'].values,
        'gw_names': gw_beta['variable'].values if 'variable' in gw_beta.columns else None,
        'shortage': shortage_beta['beta'].values,
        'gcpi': gcpi_beta['beta'].values,
        'cf1': cf1_beta['beta'].values,
        'cf10': cf10_beta['beta'].values
    }


def calculate_shock_magnitudes(table_q4_data, overrides=None):
    """
    Calculate shock magnitudes (1 standard deviation) from data.

    IMPORTANT: CU shock magnitude differs significantly between specifications:
    - Level CU (USE_LOG_CU_WAGES=False): std dev ~ 0.03-0.05
    - Log CU (USE_LOG_CU_WAGES=True): std dev ~ 0.005-0.01

    Returns dict of shock magnitudes.
    """
    overrides = overrides or {}

    shocks = {}

    # Calculate from data or use override
    def get_shock(name, col, default):
        if overrides.get(name) is not None:
            return overrides[name]
        if col in table_q4_data.columns:
            return np.nanstd(table_q4_data[col])
        return default

    shocks['grpe'] = get_shock('grpe', 'grpe', 10.0)
    shocks['grpf'] = get_shock('grpf', 'grpf', 5.0)
    shocks['vu'] = get_shock('vu', 'vu', 0.3)
    shocks['shortage'] = get_shock('shortage', 'shortage', 10.0)
    shocks['gscpi'] = get_shock('gscpi', 'gscpi', 1.0)

    # CU shock - use 'cu' column which is already detrended in the correct form
    shocks['cu'] = get_shock('cu', 'cu', 0.04 if not USE_LOG_CU_WAGES else 0.01)

    # NGDPPOT shock - calculate std dev of log(NGDPPOT) growth or use default
    # Typical quarterly growth of potential GDP is ~0.5-1% annualized
    # In log terms, a 1 std dev shock is roughly 0.01-0.02 (1-2%)
    if overrides.get('ngdppot') is not None:
        shocks['ngdppot'] = overrides['ngdppot']
    elif 'log_ngdppot' in table_q4_data.columns:
        # Use std dev of log level changes
        log_ngdppot_diff = table_q4_data['log_ngdppot'].diff()
        shocks['ngdppot'] = np.nanstd(log_ngdppot_diff) if log_ngdppot_diff.notna().any() else 0.01
    else:
        shocks['ngdppot'] = 0.01  # Default: 1% shock to potential GDP

    return shocks


def print_shock_magnitudes(shock_vals):
    """Print shock magnitudes for verification."""
    print("\n" + "=" * 60)
    print("SHOCK MAGNITUDES (1 standard deviation)")
    print("=" * 60)
    print(f"  Energy (grpe):     {shock_vals['grpe']:8.4f}  (annualized %)")
    print(f"  Food (grpf):       {shock_vals['grpf']:8.4f}  (annualized %)")
    print(f"  V/U ratio:         {shock_vals['vu']:8.4f}  (ratio)")
    print(f"  Shortage:          {shock_vals['shortage']:8.4f}  (index)")
    print(f"  GSCPI:             {shock_vals['gscpi']:8.4f}  (std dev)")
    print(f"  Capacity util:     {shock_vals['cu']:8.4f}  ({'log units' if USE_LOG_CU_WAGES else 'level/100'})")
    print(f"  Potential GDP:     {shock_vals['ngdppot']:8.4f}  (log deviation)")
    print("=" * 60)

    if USE_LOG_CU_WAGES and shock_vals['cu'] > 0.05:
        print("\n  WARNING: CU shock seems large for log specification!")
        print("           Expected ~0.005-0.01 for log CU")
    elif not USE_LOG_CU_WAGES and shock_vals['cu'] < 0.01:
        print("\n  WARNING: CU shock seems small for level specification!")
        print("           Expected ~0.03-0.05 for level CU")


# %%
def irfs_new_model(data, shock_vals, shocks, rho, coef_path):
    """
    Run impulse response functions for the modified model.

    IMPORTANT: Excess demand is computed ENDOGENOUSLY from simulated wage levels.
    The simulation order is: Wages -> Excess Demand -> Shortage -> Prices

    Parameters:
    -----------
    data : DataFrame
        Historical data (used for reference)
    shock_vals : dict
        Dictionary of shock magnitudes: {shock_name: value}
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
    add_ngdppot_shock = shocks.get('ngdppot', False)

    # Extract persistence parameters
    rho_grpe = rho.get('grpe', 0.0)
    rho_grpf = rho.get('grpf', 0.0)
    rho_vu = rho.get('vu', 1.0)
    rho_shortage = rho.get('shortage', 0.0)
    rho_gscpi = rho.get('gscpi', 0.0)
    rho_gcu = rho.get('gcu', 0.0)
    rho_ngdppot = rho.get('ngdppot', 0.0)

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
    log_w_simul = np.zeros(timesteps)
    log_tcu_simul = np.zeros(timesteps)  # Track TCU deviation for excess demand
    log_ngdppot_simul = np.zeros(timesteps)  # Track NGDPPOT deviation for excess demand
    ed_simul = np.zeros(timesteps)
    raw_ed_simul = np.zeros(timesteps)

    # Exogenous shock series
    grpe_shock_series = np.zeros(timesteps)
    grpf_shock_series = np.zeros(timesteps)
    vu_shock_series = np.zeros(timesteps)
    gscpi_shock_series = np.zeros(timesteps)
    gcu_shock_series = np.zeros(timesteps)
    ngdppot_shock_series = np.zeros(timesteps)

    # Get shock magnitudes
    shock_val_grpe = shock_vals['grpe']
    shock_val_grpf = shock_vals['grpf']
    shock_val_vu = shock_vals['vu']
    shock_val_shortage = shock_vals['shortage']
    shock_val_gscpi = shock_vals['gscpi']
    shock_val_cu = shock_vals['cu']
    shock_val_ngdppot = shock_vals['ngdppot']

    # =========================================================================
    # COEFFICIENT INDICES
    # The structure depends on USE_CONTEMP_CU flag
    # =========================================================================
    # Wage equation structure:
    # [const, L1-L4 gw, L1-L4 cf1, L1 magpty, L1-L4 vu, L1-L4 diffcpicf, cu_lags, dummies]
    #
    # With USE_CONTEMP_CU=True:  cu has L0-L4 (5 terms), starts at index 18
    # With USE_CONTEMP_CU=False: cu has L1-L4 (4 terms), starts at index 18
    #
    # COVID dummies only present in full sample (not pre-COVID)

    # Base indices (constant through diffcpicf)
    idx_const = 0
    idx_gw = slice(1, 5)       # gw L1-L4
    idx_cf1 = slice(5, 9)      # cf1 L1-L4
    idx_magpty = 9             # magpty L1
    idx_vu = slice(10, 14)     # vu L1-L4
    idx_diffcpicf = slice(14, 18)  # diffcpicf L1-L4

    # CU indices depend on specification
    if USE_CONTEMP_CU:
        idx_cu_start = 18
        idx_cu_count = 5  # L0-L4
    else:
        idx_cu_start = 18
        idx_cu_count = 4  # L1-L4

    # Run IRF simulation
    for t in range(4, timesteps):

        # Apply shocks at t=4 (first simulation period)
        shock_grpe = shock_val_grpe if (add_grpe_shock and t == 4) else 0
        shock_grpf = shock_val_grpf if (add_grpf_shock and t == 4) else 0
        shock_vu = shock_val_vu if (add_vu_shock and t == 4) else 0
        shock_gscpi = shock_val_gscpi if (add_gscpi_shock and t == 4) else 0
        shock_cu = -shock_val_cu if (add_gcu_shock and t == 4) else 0  # Negative shock
        shock_ngdppot = -shock_val_ngdppot if (add_ngdppot_shock and t == 4) else 0  # Negative shock

        # Update exogenous shock series with persistence
        grpe_shock_series[t] = rho_grpe * grpe_shock_series[t-1] + shock_grpe
        grpf_shock_series[t] = rho_grpf * grpf_shock_series[t-1] + shock_grpf
        vu_shock_series[t] = rho_vu * vu_shock_series[t-1] + shock_vu
        gscpi_shock_series[t] = rho_gscpi * gscpi_shock_series[t-1] + shock_gscpi
        gcu_shock_series[t] = rho_gcu * gcu_shock_series[t-1] + shock_cu
        ngdppot_shock_series[t] = rho_ngdppot * ngdppot_shock_series[t-1] + shock_ngdppot

        # =====================================================================
        # STEP 1: WAGE EQUATION
        # NOTE: Exclude constant for IRF (measuring deviations from steady state)
        # =====================================================================

        # Base components (same for all specifications)
        gw_val = (
            gw_beta[1] * gw_simul[t-1] +
            gw_beta[2] * gw_simul[t-2] +
            gw_beta[3] * gw_simul[t-3] +
            gw_beta[4] * gw_simul[t-4] +
            gw_beta[5] * cf1_simul[t-1] +
            gw_beta[6] * cf1_simul[t-2] +
            gw_beta[7] * cf1_simul[t-3] +
            gw_beta[8] * cf1_simul[t-4] +
            # gw_beta[9] * magpty[t-1] is zero in IRF
            gw_beta[10] * vu_shock_series[t-1] +
            gw_beta[11] * vu_shock_series[t-2] +
            gw_beta[12] * vu_shock_series[t-3] +
            gw_beta[13] * vu_shock_series[t-4] +
            gw_beta[14] * diffcpicf_simul[t-1] +
            gw_beta[15] * diffcpicf_simul[t-2] +
            gw_beta[16] * diffcpicf_simul[t-3] +
            gw_beta[17] * diffcpicf_simul[t-4]
        )

        # Add CU terms based on specification
        if USE_CONTEMP_CU:
            # CU L0-L4 (5 terms starting at index 18)
            gw_val += (
                gw_beta[18] * gcu_shock_series[t] +     # L0 (contemporaneous)
                gw_beta[19] * gcu_shock_series[t-1] +   # L1
                gw_beta[20] * gcu_shock_series[t-2] +   # L2
                gw_beta[21] * gcu_shock_series[t-3] +   # L3
                gw_beta[22] * gcu_shock_series[t-4]     # L4
            )
        else:
            # CU L1-L4 (4 terms starting at index 18)
            gw_val += (
                gw_beta[18] * gcu_shock_series[t-1] +   # L1
                gw_beta[19] * gcu_shock_series[t-2] +   # L2
                gw_beta[20] * gcu_shock_series[t-3] +   # L3
                gw_beta[21] * gcu_shock_series[t-4]     # L4
            )

        gw_simul[t] = gw_val

        # =====================================================================
        # STEP 2: UPDATE WAGE LEVEL AND COMPUTE EXCESS DEMAND (endogenous)
        #
        # Excess demand = log(W) - log(NGDPPOT) - log(TCU/100)
        #
        # In deviations from steady state:
        # - log_w_simul tracks wage level deviation
        # - log_tcu_simul tracks TCU deviation (from CU shocks)
        # - NGDPPOT deviation = 0 (not shocked)
        #
        # For CU shocks, we need to account for the DIRECT effect on ED:
        # - Higher TCU means more capacity → LOWER excess demand
        # - This partially offsets the wage effect from CU → wages → ED
        #
        # Conversion from detrended CU to log(TCU/100) deviation:
        # - Log CU spec: cu = log(TCU/100) - trend, so cu ≈ d(log(TCU/100))
        # - Level CU spec: cu = (TCU - trend)/100
        #   Near steady state (TCU≈80): d(log(TCU/100)) ≈ cu / 0.8 ≈ 1.25 * cu
        #   For simplicity, we use a first-order approximation: log_tcu ≈ cu
        # =====================================================================
        log_w_simul[t] = log_w_simul[t-1] + gw_simul[t] / 400.0

        # Update log TCU deviation based on CU shock
        # For log CU: gcu_shock_series directly represents log(TCU/100) deviation
        # For level CU: approximate conversion (first-order Taylor expansion)
        if USE_LOG_CU_WAGES:
            # Log CU: cu = log(TCU/100) - log(TCU_trend/100), so deviation is direct
            log_tcu_simul[t] = gcu_shock_series[t]
        else:
            # Level CU: cu = (TCU - TCU_trend) / 100
            # d(log(TCU/100)) ≈ d(TCU/100) / (TCU_0/100) ≈ cu / 0.8
            # Using 0.8 as typical TCU level (80%)
            log_tcu_simul[t] = gcu_shock_series[t] / 0.8

        # Update log NGDPPOT deviation (positive shock = higher potential GDP)
        # NGDPPOT shock is in log units directly
        log_ngdppot_simul[t] = ngdppot_shock_series[t]

        # Raw excess demand = log(W) - log(NGDPPOT) - log(TCU/100)
        # In deviations from steady state:
        # ED_deviation = log_w_deviation - log_ngdppot_deviation - log_tcu_deviation
        #
        # Note: A positive NGDPPOT shock DECREASES excess demand (more capacity)
        #       A positive TCU shock also DECREASES excess demand (more utilization)
        #       A positive wage shock INCREASES excess demand
        raw_ed_simul[t] = log_w_simul[t] - log_ngdppot_simul[t] - log_tcu_simul[t]

        # Compute rolling trend (40-quarter rolling mean)
        lookback = min(40, t + 1)
        rolling_trend = np.mean(raw_ed_simul[t-lookback+1:t+1])

        # Detrended excess demand (if using detrending)
        if USE_DETRENDED_EXCESS_DEMAND:
            ed_simul[t] = raw_ed_simul[t] - rolling_trend
        else:
            ed_simul[t] = raw_ed_simul[t]

        # =====================================================================
        # STEP 3: SHORTAGE EQUATION (uses ENDOGENOUS excess demand)
        # Coefficients: [const, L1-L4 shortage, excess_demand L0-L4, gscpi L0-L4]
        # =====================================================================
        if add_shortage_shock and t == 4:
            # Direct shock to shortage (bypasses the shortage equation)
            shortage_simul[t] = shock_val_shortage
        else:
            shortage_simul[t] = (
                shortage_beta[1] * shortage_simul[t-1] +
                shortage_beta[2] * shortage_simul[t-2] +
                shortage_beta[3] * shortage_simul[t-3] +
                shortage_beta[4] * shortage_simul[t-4] +
                shortage_beta[5] * ed_simul[t] +
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
            gcpi_beta[21] * shortage_simul[t] +
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
        'ed_simul': ed_simul,
        'raw_ed_simul': raw_ed_simul,
        'log_w_simul': log_w_simul,
        'log_tcu_simul': log_tcu_simul,
        'log_ngdppot_simul': log_ngdppot_simul,
        'cf1_simul': cf1_simul,
        'cf10_simul': cf10_simul,
        'diffcpicf_simul': diffcpicf_simul,
        'grpe_shock_series': grpe_shock_series,
        'grpf_shock_series': grpf_shock_series,
        'vu_shock_series': vu_shock_series,
        'gscpi_shock_series': gscpi_shock_series,
        'gcu_shock_series': gcu_shock_series,
        'ngdppot_shock_series': ngdppot_shock_series
    })

    return results


# %%
# Calculate shock magnitudes
shock_vals = calculate_shock_magnitudes(table_q4_data, SHOCK_OVERRIDES)
print_shock_magnitudes(shock_vals)

# %%
print("\n" + "=" * 80)
print("RUNNING IRF SIMULATIONS")
print("=" * 80)

# Original BB shocks
print("\n--- Original BB Shocks ---")

print("\nEnergy prices shock:")
results_energy = irfs_new_model(
    data, shock_vals,
    shocks={'grpe': True},
    rho={'grpe': 0.0},
    coef_path=coef_path
)

print("\nFood prices shock:")
results_food = irfs_new_model(
    data, shock_vals,
    shocks={'grpf': True},
    rho={'grpf': 0.0},
    coef_path=coef_path
)

print("\nV/U shock (persistent):")
results_vu = irfs_new_model(
    data, shock_vals,
    shocks={'vu': True},
    rho={'vu': 1.0},  # Permanent shock
    coef_path=coef_path
)

print("\nShortage shock (direct):")
results_shortage = irfs_new_model(
    data, shock_vals,
    shocks={'shortage': True},
    rho={'shortage': 0.0},
    coef_path=coef_path
)

# NEW model-specific shocks
print("\n--- New Model Shocks ---")

print("\nGSCPI shock:")
results_gscpi = irfs_new_model(
    data, shock_vals,
    shocks={'gscpi': True},
    rho={'gscpi': 0.0},
    coef_path=coef_path
)

print("\nCapacity utilization shock:")
results_gcu = irfs_new_model(
    data, shock_vals,
    shocks={'gcu': True},
    rho={'gcu': 0.0},
    coef_path=coef_path
)

print("\nPotential GDP shock:")
results_ngdppot = irfs_new_model(
    data, shock_vals,
    shocks={'ngdppot': True},
    rho={'ngdppot': 0.0},
    coef_path=coef_path
)

# Persistent versions
print("\n--- Persistent Shocks ---")

print("\nGSCPI shock (persistent):")
results_gscpi_persistent = irfs_new_model(
    data, shock_vals,
    shocks={'gscpi': True},
    rho={'gscpi': 0.9},
    coef_path=coef_path
)

print("\nCapacity utilization shock (persistent):")
results_gcu_persistent = irfs_new_model(
    data, shock_vals,
    shocks={'gcu': True},
    rho={'gcu': 0.9},
    coef_path=coef_path
)

print("\nPotential GDP shock (persistent):")
results_ngdppot_persistent = irfs_new_model(
    data, shock_vals,
    shocks={'ngdppot': True},
    rho={'ngdppot': 0.9},
    coef_path=coef_path
)

# Additional shock: Excess demand via V/U + CU combined
print("\nExcess demand shock (V/U + CU combined):")
results_excess_demand = irfs_new_model(
    data, shock_vals,
    shocks={'vu': True, 'gcu': True},
    rho={'vu': 1.0, 'gcu': 0.9},
    coef_path=coef_path
)


# %%
print("\n" + "=" * 80)
print("EXPORTING RESULTS")
print("=" * 80)

# Export to Excel
results_energy.to_excel(output_dir / 'results_energy.xlsx', index=False)
results_food.to_excel(output_dir / 'results_food.xlsx', index=False)
results_vu.to_excel(output_dir / 'results_vu.xlsx', index=False)
results_shortage.to_excel(output_dir / 'results_shortage.xlsx', index=False)
results_gscpi.to_excel(output_dir / 'results_gscpi.xlsx', index=False)
results_gcu.to_excel(output_dir / 'results_gcu.xlsx', index=False)
results_ngdppot.to_excel(output_dir / 'results_ngdppot.xlsx', index=False)
results_gscpi_persistent.to_excel(output_dir / 'results_gscpi_persistent.xlsx', index=False)
results_gcu_persistent.to_excel(output_dir / 'results_gcu_persistent.xlsx', index=False)
results_ngdppot_persistent.to_excel(output_dir / 'results_ngdppot_persistent.xlsx', index=False)
results_excess_demand.to_excel(output_dir / 'results_excess_demand.xlsx', index=False)

# Save configuration for reference
config_df = pd.DataFrame({
    'Parameter': [
        'USE_PRE_COVID_SAMPLE',
        'USE_LOG_CU_WAGES',
        'USE_CONTEMP_CU',
        'USE_DETRENDED_EXCESS_DEMAND',
        'shock_grpe',
        'shock_grpf',
        'shock_vu',
        'shock_shortage',
        'shock_gscpi',
        'shock_cu',
        'shock_ngdppot',
    ],
    'Value': [
        USE_PRE_COVID_SAMPLE,
        USE_LOG_CU_WAGES,
        USE_CONTEMP_CU,
        USE_DETRENDED_EXCESS_DEMAND,
        shock_vals['grpe'],
        shock_vals['grpf'],
        shock_vals['vu'],
        shock_vals['shortage'],
        shock_vals['gscpi'],
        shock_vals['cu'],
        shock_vals['ngdppot'],
    ]
})
config_df.to_excel(output_dir / 'irf_config.xlsx', index=False)

print(f"\nResults saved to: {output_dir}")
print("  Original BB shocks:")
print("    - results_energy.xlsx")
print("    - results_food.xlsx")
print("    - results_vu.xlsx")
print("    - results_shortage.xlsx")
print("  New model shocks:")
print("    - results_gscpi.xlsx")
print("    - results_gcu.xlsx")
print("    - results_ngdppot.xlsx")
print("  Persistent shocks:")
print("    - results_gscpi_persistent.xlsx")
print("    - results_gcu_persistent.xlsx")
print("    - results_ngdppot_persistent.xlsx")
print("  Combined shocks:")
print("    - results_excess_demand.xlsx")
print("  Configuration:")
print("    - irf_config.xlsx")


# %%
# Print summary of IRF responses
print("\n" + "=" * 80)
print("IRF SUMMARY")
print("=" * 80)

print(f"\nOriginal BB Shocks (Peak inflation response):")
print(f"  Energy shock:     Peak gcpi = {results_energy['gcpi_simul'].max():.4f} at period {results_energy['gcpi_simul'].argmax() + 1}")
print(f"  Food shock:       Peak gcpi = {results_food['gcpi_simul'].max():.4f} at period {results_food['gcpi_simul'].argmax() + 1}")
print(f"  V/U shock:        Peak gcpi = {results_vu['gcpi_simul'].max():.4f} at period {results_vu['gcpi_simul'].argmax() + 1}")
print(f"  Shortage shock:   Peak gcpi = {results_shortage['gcpi_simul'].max():.4f} at period {results_shortage['gcpi_simul'].argmax() + 1}")

print(f"\nNew Model Shocks (one-time):")
print(f"  GSCPI:            Peak gcpi = {results_gscpi['gcpi_simul'].max():.4f} at period {results_gscpi['gcpi_simul'].argmax() + 1}")
print(f"  Capacity util:    Peak gcpi = {results_gcu['gcpi_simul'].max():.4f} at period {results_gcu['gcpi_simul'].argmax() + 1}")
print(f"  Potential GDP:    Peak gcpi = {results_ngdppot['gcpi_simul'].min():.4f} at period {results_ngdppot['gcpi_simul'].argmin() + 1}")

print(f"\nNew Model Shocks (persistent, rho=0.9):")
print(f"  GSCPI:            Peak gcpi = {results_gscpi_persistent['gcpi_simul'].max():.4f} at period {results_gscpi_persistent['gcpi_simul'].argmax() + 1}")
print(f"  Capacity util:    Peak gcpi = {results_gcu_persistent['gcpi_simul'].max():.4f} at period {results_gcu_persistent['gcpi_simul'].argmax() + 1}")
print(f"  Potential GDP:    Min gcpi = {results_ngdppot_persistent['gcpi_simul'].min():.4f} at period {results_ngdppot_persistent['gcpi_simul'].argmin() + 1}")

# Show the V/U -> ED -> Shortage feedback
print(f"\nV/U Shock - Endogenous Feedback Loop:")
print(f"  Peak wage growth (gw):     {results_vu['gw_simul'].max():.4f} at period {results_vu['gw_simul'].argmax() + 1}")
print(f"  Peak log wage level:       {results_vu['log_w_simul'].max():.4f} at period {results_vu['log_w_simul'].argmax() + 1}")
print(f"  Peak excess demand (ed):   {results_vu['ed_simul'].max():.4f} at period {results_vu['ed_simul'].argmax() + 1}")
print(f"  Peak shortage:             {results_vu['shortage_simul'].max():.4f} at period {results_vu['shortage_simul'].argmax() + 1}")
print(f"  Peak inflation (gcpi):     {results_vu['gcpi_simul'].max():.4f} at period {results_vu['gcpi_simul'].argmax() + 1}")

print(f"\nCapacity Utilization Shock - Endogenous Feedback Loop:")
print(f"  Peak wage growth (gw):     {results_gcu['gw_simul'].max():.4f} at period {results_gcu['gw_simul'].argmax() + 1}")
print(f"  Peak log wage level:       {results_gcu['log_w_simul'].max():.4f} at period {results_gcu['log_w_simul'].argmax() + 1}")
print(f"  Peak log TCU deviation:    {results_gcu['log_tcu_simul'].max():.4f} at period {results_gcu['log_tcu_simul'].argmax() + 1}")
print(f"  Peak excess demand (ed):   {results_gcu['ed_simul'].max():.4f} at period {results_gcu['ed_simul'].argmax() + 1}")
print(f"  Peak shortage:             {results_gcu['shortage_simul'].max():.4f} at period {results_gcu['shortage_simul'].argmax() + 1}")
print(f"  Peak inflation (gcpi):     {results_gcu['gcpi_simul'].max():.4f} at period {results_gcu['gcpi_simul'].argmax() + 1}")

print(f"\nPotential GDP Shock - Endogenous Feedback Loop:")
print(f"  (Note: Positive NGDPPOT shock = more capacity = lower excess demand = lower inflation)")
print(f"  Log NGDPPOT deviation:     {results_ngdppot['log_ngdppot_simul'].max():.4f} at period {results_ngdppot['log_ngdppot_simul'].argmax() + 1}")
print(f"  Min excess demand (ed):    {results_ngdppot['ed_simul'].min():.4f} at period {results_ngdppot['ed_simul'].argmin() + 1}")
print(f"  Min shortage:              {results_ngdppot['shortage_simul'].min():.4f} at period {results_ngdppot['shortage_simul'].argmin() + 1}")
print(f"  Min inflation (gcpi):      {results_ngdppot['gcpi_simul'].min():.4f} at period {results_ngdppot['gcpi_simul'].argmin() + 1}")

print("\n" + "=" * 80)
print("IRF SIMULATIONS COMPLETE!")
print("=" * 80)
print("\n")

# %%
