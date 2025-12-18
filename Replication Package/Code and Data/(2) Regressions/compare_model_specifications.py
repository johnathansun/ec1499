#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare Out-of-Sample Predictive Performance Across Model Specifications
Liang & Sun (2025)

This script computes RMSE for out-of-sample predictions (2020+) across different
model specifications and outputs a comparison table.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# %%
#*******************************************************************************
# CONFIGURATION
#*******************************************************************************

BASE_DIR = Path("/Users/johnathansun/Documents/ec1499/Replication Package/Code and Data/(2) Regressions")
OUTPUT_FILE = BASE_DIR / "model_comparison_rmse.xlsx"

# Define model specifications to compare
# Format: (folder_name, display_name, suffix, file_pattern)
# file_pattern: None = use default, or specify custom pattern
MODEL_SPECS = [
    # Original Bernanke-Blanchard model
    ("Old Output/Output Data (Pre Covid Sample)", "BB Original", "", "eq_simulations_data_restricted.xls"),
    # New model specifications (Pre-COVID sample for out-of-sample 2020+)
    ("Output Data (New Pre Covid)", "New Base", "_pre_covid", None),
    ("Output Data (New Pre Covid Contemp CU)", "New +CU(0)", "_pre_covid", None),
    ("Output Data (New Pre Covid Log CU)", "New +LogCU", "_pre_covid", None),
    ("Output Data (New Pre Covid Log CU Contemp CU)", "New +Log+CU(0)", "_pre_covid", None),
    ("Output Data (New Pre Covid Detrended ED)", "New +DetrendED", "_pre_covid", None),
]

# Variables to compare (actual_col, pred_col, display_name)
VARIABLES = [
    ("gw", "gwf1", "Wage Growth"),
    ("gcpi", "gcpif", "Inflation"),
    ("cf1", "cf1f", "Short-run Exp."),
    ("cf10", "cf10f", "Long-run Exp."),
    ("shortage", "shortagef", "Shortage"),
]

# Out-of-sample period
OOS_START = "2020-01-01"

#*******************************************************************************
# HELPER FUNCTIONS
#*******************************************************************************

def rmse(actual, pred):
    """Compute Root Mean Squared Error."""
    valid = actual.notna() & pred.notna()
    if valid.sum() == 0:
        return np.nan
    return np.sqrt(np.mean((actual[valid] - pred[valid])**2))


def mae(actual, pred):
    """Compute Mean Absolute Error."""
    valid = actual.notna() & pred.notna()
    if valid.sum() == 0:
        return np.nan
    return np.mean(np.abs(actual[valid] - pred[valid]))


def mape(actual, pred):
    """Compute Mean Absolute Percentage Error."""
    valid = actual.notna() & pred.notna() & (actual != 0)
    if valid.sum() == 0:
        return np.nan
    return np.mean(np.abs((actual[valid] - pred[valid]) / actual[valid])) * 100


def load_model_data(folder_name, suffix, file_pattern=None):
    """Load simulation data for a model specification."""
    folder = BASE_DIR / folder_name
    if not folder.exists():
        return None

    # If custom file pattern provided, use it
    if file_pattern:
        file_path = folder / file_pattern
        if file_path.exists():
            df = pd.read_excel(file_path)
            df['period'] = pd.to_datetime(df['period'])
            return df
        return None

    # Try different file naming patterns
    patterns = [
        f"eq_simulations_data_new_model{suffix}.xlsx",
        f"eq_simulations_data{suffix}.xlsx",
    ]

    for pattern in patterns:
        file_path = folder / pattern
        if file_path.exists():
            df = pd.read_excel(file_path)
            df['period'] = pd.to_datetime(df['period'])
            return df

    return None


#*******************************************************************************
# MAIN ANALYSIS
#*******************************************************************************

print("="*80)
print("COMPARING OUT-OF-SAMPLE PREDICTIVE PERFORMANCE")
print("="*80)
print(f"\nOut-of-sample period: {OOS_START} onwards")
print(f"Comparing {len(MODEL_SPECS)} model specifications\n")

# Check which folders exist
available_specs = []
for spec in MODEL_SPECS:
    folder, name, suffix, file_pattern = spec
    folder_path = BASE_DIR / folder
    if folder_path.exists():
        available_specs.append(spec)
        print(f"  [OK] {name}: {folder}")
    else:
        print(f"  [--] {name}: {folder} (not found)")

print("\n" + "-"*80)

# Load all data
model_data = {}
for folder, name, suffix, file_pattern in available_specs:
    df = load_model_data(folder, suffix, file_pattern)
    if df is not None:
        model_data[name] = df
        print(f"Loaded {name}: {len(df)} observations")

# Filter to out-of-sample period
oos_data = {}
for name, df in model_data.items():
    oos_df = df[df['period'] >= OOS_START].copy()
    oos_data[name] = oos_df
    print(f"  {name} OOS: {len(oos_df)} observations")

print("\n" + "="*80)
print("RMSE COMPARISON TABLE")
print("="*80)

# Compute RMSE for each model and variable
results = []
for var_actual, var_pred, var_name in VARIABLES:
    row = {"Variable": var_name}
    for name in model_data.keys():
        df = oos_data[name]
        if var_actual in df.columns and var_pred in df.columns:
            row[name] = rmse(df[var_actual], df[var_pred])
        else:
            row[name] = np.nan
    results.append(row)

# Create DataFrame
rmse_df = pd.DataFrame(results)
rmse_df = rmse_df.set_index("Variable")

# Print table
print("\nRMSE (Root Mean Squared Error):")
print(rmse_df.round(4).to_string())

# Compute relative performance (% change from BB Original)
baseline_col = "BB Original" if "BB Original" in rmse_df.columns else rmse_df.columns[0]
print("\n" + "-"*80)
print(f"\nRelative Performance (% change vs {baseline_col}):")
rel_df = rmse_df.copy()
for col in rel_df.columns:
    if col != baseline_col:
        rel_df[col] = ((rel_df[col] - rel_df[baseline_col]) / rel_df[baseline_col] * 100)
rel_df[baseline_col] = 0.0
print(rel_df.round(2).to_string())
print(f"\n(Negative = better than {baseline_col})")

# Also compute MAE
print("\n" + "="*80)
print("MAE COMPARISON TABLE")
print("="*80)

mae_results = []
for var_actual, var_pred, var_name in VARIABLES:
    row = {"Variable": var_name}
    for name in model_data.keys():
        df = oos_data[name]
        if var_actual in df.columns and var_pred in df.columns:
            row[name] = mae(df[var_actual], df[var_pred])
        else:
            row[name] = np.nan
    mae_results.append(row)

mae_df = pd.DataFrame(mae_results)
mae_df = mae_df.set_index("Variable")
print("\nMAE (Mean Absolute Error):")
print(mae_df.round(4).to_string())

# Save to Excel
print("\n" + "="*80)
print("SAVING RESULTS")
print("="*80)

with pd.ExcelWriter(OUTPUT_FILE, engine='openpyxl') as writer:
    rmse_df.to_excel(writer, sheet_name='RMSE')
    mae_df.to_excel(writer, sheet_name='MAE')

    # Add relative performance
    rel_rmse = rmse_df.copy()
    for col in rel_rmse.columns:
        if col != baseline_col:
            rel_rmse[col] = ((rel_rmse[col] - rel_rmse[baseline_col]) / rel_rmse[baseline_col] * 100)
    rel_rmse[baseline_col] = 0.0
    rel_rmse.to_excel(writer, sheet_name='RMSE Relative (%)')

    rel_mae = mae_df.copy()
    for col in rel_mae.columns:
        if col != baseline_col:
            rel_mae[col] = ((rel_mae[col] - rel_mae[baseline_col]) / rel_mae[baseline_col] * 100)
    rel_mae[baseline_col] = 0.0
    rel_mae.to_excel(writer, sheet_name='MAE Relative (%)')

print(f"\nSaved comparison tables to: {OUTPUT_FILE}")

# Print summary
print("\n" + "="*80)
print("SUMMARY: BEST MODEL BY VARIABLE (Lowest RMSE)")
print("="*80)

for var_name in rmse_df.index:
    row = rmse_df.loc[var_name]
    best_model = row.idxmin()
    best_rmse = row.min()
    print(f"  {var_name:<20}: {best_model} (RMSE = {best_rmse:.4f})")

# Note about model differences
print("\n" + "-"*80)
print("NOTES:")
print("  - BB Original: Original Bernanke-Blanchard model (no CU, exogenous shortage)")
print("  - New Base: Adds endogenous shortage equation, no CU in wage equation")
print("  - New +CU(0): Adds contemporaneous CU to wage equation")
print("  - New +LogCU: Uses log(CU) instead of level CU")
print("  - New +Log+CU(0): Log CU with contemporaneous term")
print("")
print("  Shortage predictions only available for New models (BB treats shortage as exogenous)")

# Create a nicely formatted comparison table
print("\n" + "="*80)
print("WAGE GROWTH PREDICTION COMPARISON (Out-of-Sample 2020+)")
print("="*80)

wage_comparison = pd.DataFrame({
    'Model': rmse_df.columns.tolist(),
    'RMSE': rmse_df.loc['Wage Growth'].values,
    'MAE': mae_df.loc['Wage Growth'].values,
})
if baseline_col in rmse_df.columns:
    base_rmse = rmse_df.loc['Wage Growth', baseline_col]
    wage_comparison[f'% vs {baseline_col}'] = ((wage_comparison['RMSE'] - base_rmse) / base_rmse * 100).round(2)

print(wage_comparison.to_string(index=False))

# Inflation comparison
print("\n" + "="*80)
print("INFLATION PREDICTION COMPARISON (Out-of-Sample 2020+)")
print("="*80)

infl_comparison = pd.DataFrame({
    'Model': rmse_df.columns.tolist(),
    'RMSE': rmse_df.loc['Inflation'].values,
    'MAE': mae_df.loc['Inflation'].values,
})
if baseline_col in rmse_df.columns:
    base_rmse = rmse_df.loc['Inflation', baseline_col]
    infl_comparison[f'% vs {baseline_col}'] = ((infl_comparison['RMSE'] - base_rmse) / base_rmse * 100).round(2)

print(infl_comparison.to_string(index=False))

print("\n" + "="*80)
print("Done!")
print("="*80)
# %%
