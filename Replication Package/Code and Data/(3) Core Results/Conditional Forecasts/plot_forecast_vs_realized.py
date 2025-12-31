# %%
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Forecast vs Realized Data Comparison
Liang & Sun (2025)

This script compares conditional forecasts from both the BB and New Model
against realized data.

DATA INPUT OPTIONS:
1. Automatically uses Regression_Data.xlsx (up to Q2 2023)
2. Add more recent data by filling in realized_data_raw.csv with raw FRED series
   - The script will automatically calculate gcpi, gw, etc. from raw data

To update with recent data:
1. Download raw series from FRED (CPIAUCSL, ECIWAG, etc.)
2. Paste into realized_data_raw.csv
3. Run this script
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# %%
#****************************CONFIGURATION**************************************

USE_PRE_COVID_SAMPLE = False
USE_LOG_CU_WAGES = False
USE_CONTEMP_CU = False
USE_DETRENDED_EXCESS_DEMAND = True

# Set to True to use ONLY the data in realized_data_raw.csv (ignore Regression_Data.xlsx)
USE_RAW_CSV_ONLY = True

#****************************PATH CONFIGURATION*********************************

BASE_DIR = Path("/Users/johnathansun/Documents/ec1499/Replication Package/Code and Data")

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

# Input paths
new_model_dir = BASE_DIR / "(3) Core Results/Conditional Forecasts/Output Data Python (New Model)" / SPEC_DIR_NAME
bb_model_dir = BASE_DIR / "(3) Core Results/Conditional Forecasts/Output Data Python"
regression_data_path = BASE_DIR / "(1) Data/Public Data/Regression_Data.xlsx"
raw_csv_path = BASE_DIR / "(3) Core Results/Conditional Forecasts/realized_data_raw.csv"

# Output path
output_dir = BASE_DIR / "(3) Core Results/Conditional Forecasts/Figures Python (New Model)" / SPEC_DIR_NAME
output_dir.mkdir(parents=True, exist_ok=True)

#*******************************************************************************
# DATA PROCESSING FUNCTIONS (same as regression_new_model_refactored.py)
#*******************************************************************************

def parse_quarter(date_str):
    """Parse 'YYYY QN' format to datetime."""
    if pd.isna(date_str):
        return pd.NaT
    date_str = str(date_str).strip()
    try:
        year, q = date_str.split(' Q')
        month = (int(q) - 1) * 3 + 1
        return pd.Timestamp(year=int(year), month=month, day=1)
    except:
        return pd.NaT

def calculate_derived_variables(df):
    """
    Calculate derived variables from raw FRED data.
    Uses same formulas as regression_new_model_refactored.py
    """
    df = df.copy()

    # Sort by period
    df = df.sort_values('period').reset_index(drop=True)

    # CPI inflation (annualized quarterly)
    # gcpi = 400 * (log(CPIAUCSL_t) - log(CPIAUCSL_{t-1}))
    if 'CPIAUCSL' in df.columns:
        df['gcpi'] = 400 * (np.log(df['CPIAUCSL']) - np.log(df['CPIAUCSL'].shift(1)))

    # Wage growth (annualized quarterly)
    # gw = 400 * (log(ECIWAG_t) - log(ECIWAG_{t-1}))
    if 'ECIWAG' in df.columns:
        df['gw'] = 400 * (np.log(df['ECIWAG']) - np.log(df['ECIWAG'].shift(1)))

    # V/U ratio (direct from VOVERU or calculated)
    if 'VOVERU' in df.columns:
        df['vu'] = df['VOVERU']
    elif 'JTSJOL' in df.columns and 'UNEMPLOY' in df.columns:
        df['vu'] = df['JTSJOL'] / df['UNEMPLOY']

    # Inflation expectations
    if 'EXPINF1YR' in df.columns:
        df['cf1'] = df['EXPINF1YR']
    elif 'MICH' in df.columns:
        df['cf1'] = df['MICH']

    if 'EXPINF10YR' in df.columns:
        df['cf10'] = df['EXPINF10YR']
    elif 'T10YIE' in df.columns:
        df['cf10'] = df['T10YIE']

    # Shortage
    if 'SHORTAGE' in df.columns:
        df['shortage'] = df['SHORTAGE']

    # Relative energy prices (if data available)
    if 'CPIENGSL' in df.columns and 'ECIWAG' in df.columns:
        df['rpe'] = df['CPIENGSL'] / df['ECIWAG']
        df['grpe'] = 400 * (np.log(df['rpe']) - np.log(df['rpe'].shift(1)))

    # Relative food prices (if data available)
    if 'CPIUFDSL' in df.columns and 'ECIWAG' in df.columns:
        df['rpf'] = df['CPIUFDSL'] / df['ECIWAG']
        df['grpf'] = 400 * (np.log(df['rpf']) - np.log(df['rpf'].shift(1)))

    return df

def load_and_process_raw_csv(csv_path):
    """Load raw FRED data from CSV and calculate derived variables."""
    print(f"  Loading raw FRED data from: {csv_path}")

    # Read the file and find where the header row is
    with open(csv_path, 'r') as f:
        lines = f.readlines()

    # Find the header row (first line that starts with 'Date')
    header_row = None
    for i, line in enumerate(lines):
        if line.strip().startswith('Date,') or line.strip().startswith('Date\t'):
            header_row = i
            break

    if header_row is None:
        # Try reading without skipping
        df = pd.read_csv(csv_path)
    else:
        # Skip to the header row
        df = pd.read_csv(csv_path, skiprows=header_row)

    print(f"  Columns found: {df.columns.tolist()}")

    # Parse date column - handle "YYYY QN" format
    if 'Date' in df.columns:
        df['period'] = df['Date'].apply(parse_quarter)
    elif 'period' in df.columns:
        # Try to parse if it's a string like "4/1/22" or "2022-04-01"
        if not pd.api.types.is_datetime64_any_dtype(df['period']):
            try:
                df['period'] = pd.to_datetime(df['period'])
            except:
                df['period'] = df['period'].apply(parse_quarter)

    # Drop rows with invalid dates
    if 'period' in df.columns:
        df = df.dropna(subset=['period'])
    else:
        print("  Warning: No 'Date' or 'period' column found")
        return pd.DataFrame()

    # Calculate derived variables
    df = calculate_derived_variables(df)

    return df

#*********************************************************************************

print("="*80)
print("FORECAST VS REALIZED DATA COMPARISON")
print("="*80)

# %%
# Load forecast data
print("\nLoading forecast data...")

# New Model forecasts
new_model_mid = pd.read_excel(new_model_dir / 'terminal_mid.xlsx')
new_model_mid['period'] = pd.to_datetime(new_model_mid['period'])

# BB Model forecasts
bb_model_mid = pd.read_excel(bb_model_dir / 'terminal_mid.xlsx')
bb_model_mid['period'] = pd.to_datetime(bb_model_mid['period'])

print(f"  New Model forecast loaded: {len(new_model_mid)} periods")
print(f"  BB Model forecast loaded: {len(bb_model_mid)} periods")

# %%
# Load realized data
if USE_RAW_CSV_ONLY and raw_csv_path.exists():
    # Use ONLY the raw CSV data
    print("\nLoading realized data from raw CSV only...")
    try:
        realized_data = load_and_process_raw_csv(raw_csv_path)

        # Filter to valid rows (has gcpi calculated)
        realized_data = realized_data[realized_data['gcpi'].notna()].copy()

        # Select relevant columns
        cols_to_keep = ['period']
        for col in ['gcpi', 'gw', 'vu', 'cf1', 'cf10', 'shortage']:
            if col in realized_data.columns:
                cols_to_keep.append(col)
        realized_data = realized_data[cols_to_keep].copy()

        # Filter to Q2 2022 onwards
        realized_data = realized_data[realized_data['period'] >= '2022-04-01'].reset_index(drop=True)

        print(f"  Loaded {len(realized_data)} quarters from raw CSV")
    except Exception as e:
        print(f"  Error loading raw CSV: {e}")
        import traceback
        traceback.print_exc()
        realized_data = pd.DataFrame()
else:
    # Load from regression dataset (base data)
    print("\nLoading realized data from regression dataset...")

    reg_data = pd.read_excel(regression_data_path)
    reg_data['period'] = reg_data['Date'].apply(parse_quarter)
    reg_data = calculate_derived_variables(reg_data)

    # Filter to Q2 2022 onwards (forecast period)
    realized_base = reg_data[reg_data['period'] >= '2022-04-01'][
        ['period', 'gcpi', 'gw', 'vu', 'cf1', 'cf10', 'shortage']
    ].copy().reset_index(drop=True)

    print(f"  Base data from regression file: {len(realized_base)} quarters")
    print(f"  Period: {realized_base['period'].min()} to {realized_base['period'].max()}")

    # Try to load additional realized data from raw CSV
    realized_data = realized_base.copy()

    if raw_csv_path.exists():
        print("\nChecking for additional data in realized_data_raw.csv...")
        try:
            raw_df = load_and_process_raw_csv(raw_csv_path)

            # Filter to valid rows (has gcpi)
            raw_valid = raw_df[raw_df['gcpi'].notna()].copy()

            if len(raw_valid) > 0:
                # Select relevant columns
                cols_to_use = ['period']
                for col in ['gcpi', 'gw', 'vu', 'cf1', 'cf10', 'shortage']:
                    if col in raw_valid.columns:
                        cols_to_use.append(col)

                raw_subset = raw_valid[cols_to_use].copy()

                # Merge with base data (raw CSV takes precedence for overlapping dates)
                combined = pd.concat([realized_base, raw_subset], ignore_index=True)
                combined = combined.drop_duplicates(subset=['period'], keep='last')
                combined = combined.sort_values('period').reset_index(drop=True)
                realized_data = combined

                new_quarters = len(realized_data) - len(realized_base)
                print(f"  Processed raw CSV. Added/updated {new_quarters} quarters")
            else:
                print("  Raw CSV found but no additional data filled in yet")
        except Exception as e:
            print(f"  Could not process raw CSV: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"\n  Note: {raw_csv_path.name} not found. Using only regression data.")

print(f"\nFinal realized data:")
print(f"  Quarters: {len(realized_data)}")
print(f"  Range: {realized_data['period'].min()} to {realized_data['period'].max()}")

# Show the data
print("\n  Data preview:")
print(realized_data[['period', 'gcpi', 'gw', 'vu']].to_string(index=False))

# %%
# Setup plotting
colors = {
    'realized': '#000000',      # Black for actual data
    'new_model': '#009E73',      # Teal for new model
    'bb_model': '#CC79A7',       # Pink for BB model
}

plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 16,
    'axes.titlesize': 17.5,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 12,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.edgecolor': 'black',
    'axes.linewidth': 1,
    'axes.grid': True,
    'grid.color': 'lightgray',
    'grid.linestyle': '-',
    'grid.linewidth': 0.5,
})

# Filter forecasts to plotting range
plot_end = '2027-01-01'
plot_start = '2022-9-01'

new_model_plot = new_model_mid[(new_model_mid['period'] >= plot_start) &
                               (new_model_mid['period'] <= plot_end)].copy()
bb_model_plot = bb_model_mid[(bb_model_mid['period'] >= plot_start) &
                             (bb_model_mid['period'] <= plot_end)].copy()

realized_data_plot = realized_data[(realized_data['period'] >= plot_start) &
                               (realized_data['period'] <= plot_end)].copy()

# Quarter labels
def period_to_quarter_label(dt):
    return f"{dt.year} Q{(dt.month - 1) // 3 + 1}"

quarter_labels = [period_to_quarter_label(p) for p in new_model_plot['period']]
tick_positions = list(range(0, len(new_model_plot), 2))
tick_labels = [quarter_labels[i] for i in tick_positions]

# %%
# INFLATION COMPARISON: Forecast vs Realized
print("\nCreating Inflation: Forecast vs Realized plot...")

fig, ax = plt.subplots(figsize=(12, 7))

# Plot forecasts
ax.plot(new_model_plot['period'], new_model_plot['gcpi_simul'],
        color=colors['new_model'], linewidth=2, label='New Model Forecast')
ax.plot(bb_model_plot['period'], bb_model_plot['gcpi_simul'],
        color=colors['bb_model'], linewidth=2, linestyle='--', label='BB Model Forecast')

# Plot realized data
ax.plot(realized_data_plot['period'], realized_data_plot['gcpi'],
        color=colors['realized'], linewidth=2.5, marker='o', markersize=6, alpha=0.7,
        label='Realized')

ax.set_title('Inflation: Forecast vs Realized (v/u = 1.2 scenario)',
             fontsize=17.5, fontweight='normal')
ax.set_xlabel('Quarter', fontsize=16)
ax.set_ylabel('Percent (annualized)', fontsize=16)

ax.set_xticks([new_model_plot['period'].iloc[i] for i in tick_positions])
ax.set_xticklabels(tick_labels, rotation=0, ha='right')

ax.yaxis.grid(True, linestyle='-', linewidth=0.5, color='lightgray')
ax.xaxis.grid(False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=3,
          frameon=True, edgecolor='black', fancybox=False)

plt.tight_layout()
plt.savefig(output_dir / 'figure_inflation_forecast_vs_realized.png', dpi=300, bbox_inches='tight')
plt.savefig(output_dir / 'figure_inflation_forecast_vs_realized.pdf', bbox_inches='tight')
print(f"  Saved to {output_dir / 'figure_inflation_forecast_vs_realized.png'}")
plt.show()

# %%
# WAGE GROWTH COMPARISON: Forecast vs Realized
print("\nCreating Wage Growth: Forecast vs Realized plot...")

fig, ax = plt.subplots(figsize=(12, 7))

# Plot forecasts
ax.plot(new_model_plot['period'], new_model_plot['gw_simul'],
        color=colors['new_model'], linewidth=2, label='New Model Forecast')
ax.plot(bb_model_plot['period'], bb_model_plot['gw_simul'],
        color=colors['bb_model'], linewidth=2, linestyle='--', label='BB Model Forecast')

# Plot realized data
ax.plot(realized_data_plot['period'], realized_data_plot['gw'],
        color=colors['realized'], linewidth=2.5, marker='o', markersize=6, alpha=0.7,
        label='Realized')

ax.set_title('Wage Growth: Forecast vs Realized (v/u = 1.2 scenario)',
             fontsize=17.5, fontweight='normal')
ax.set_xlabel('Quarter', fontsize=16)
ax.set_ylabel('Percent (annualized)', fontsize=16)

ax.set_xticks([new_model_plot['period'].iloc[i] for i in tick_positions])
ax.set_xticklabels(tick_labels, rotation=0, ha='right')

ax.yaxis.grid(True, linestyle='-', linewidth=0.5, color='lightgray')
ax.xaxis.grid(False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=3,
          frameon=True, edgecolor='black', fancybox=False)

plt.tight_layout()
plt.savefig(output_dir / 'figure_wage_forecast_vs_realized.png', dpi=300, bbox_inches='tight')
plt.savefig(output_dir / 'figure_wage_forecast_vs_realized.pdf', bbox_inches='tight')
print(f"  Saved to {output_dir / 'figure_wage_forecast_vs_realized.png'}")
plt.show()

# %%
# COMBINED 2-PANEL FIGURE
print("\nCreating combined 2-panel figure...")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Left: Inflation
ax1 = axes[0]
ax1.plot(new_model_plot['period'], new_model_plot['gcpi_simul'],
         color=colors['new_model'], linewidth=2, label='New Model')
ax1.plot(bb_model_plot['period'], bb_model_plot['gcpi_simul'],
         color=colors['bb_model'], linewidth=2, linestyle='--', label='BB Model')
ax1.plot(realized_data['period'], realized_data['gcpi'],
         color=colors['realized'], linewidth=2.5, marker='o', markersize=5, label='Realized')

ax1.set_title('A. Inflation', fontsize=14)
ax1.set_ylabel('Percent (annualized)', fontsize=12)
ax1.set_xticks([new_model_plot['period'].iloc[i] for i in tick_positions])
ax1.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=10)
ax1.yaxis.grid(True, linestyle='-', linewidth=0.5, color='lightgray')
ax1.xaxis.grid(False)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.legend(loc='best', fontsize=10, frameon=True, edgecolor='black')

# Right: Wage Growth
ax2 = axes[1]
ax2.plot(new_model_plot['period'], new_model_plot['gw_simul'],
         color=colors['new_model'], linewidth=2, label='New Model')
ax2.plot(bb_model_plot['period'], bb_model_plot['gw_simul'],
         color=colors['bb_model'], linewidth=2, linestyle='--', label='BB Model')
ax2.plot(realized_data['period'], realized_data['gw'],
         color=colors['realized'], linewidth=2.5, marker='o', markersize=5, label='Realized')

ax2.set_title('B. Wage Growth', fontsize=14)
ax2.set_ylabel('Percent (annualized)', fontsize=12)
ax2.set_xticks([new_model_plot['period'].iloc[i] for i in tick_positions])
ax2.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=10)
ax2.yaxis.grid(True, linestyle='-', linewidth=0.5, color='lightgray')
ax2.xaxis.grid(False)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.legend(loc='best', fontsize=10, frameon=True, edgecolor='black')

plt.suptitle('Conditional Forecasts vs Realized Data (v/u = 1.2 scenario)',
             fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(output_dir / 'figure_forecast_vs_realized_combined.png', dpi=300, bbox_inches='tight')
plt.savefig(output_dir / 'figure_forecast_vs_realized_combined.pdf', bbox_inches='tight')
print(f"  Saved to {output_dir / 'figure_forecast_vs_realized_combined.png'}")
plt.show()

# %%
# Calculate and print forecast errors
print("\n" + "="*80)
print("FORECAST ERROR ANALYSIS")
print("="*80)

# Merge forecasts with realized data
merged = realized_data.merge(
    new_model_plot[['period', 'gcpi_simul', 'gw_simul']].rename(
        columns={'gcpi_simul': 'gcpi_new', 'gw_simul': 'gw_new'}),
    on='period', how='left'
)
merged = merged.merge(
    bb_model_plot[['period', 'gcpi_simul', 'gw_simul']].rename(
        columns={'gcpi_simul': 'gcpi_bb', 'gw_simul': 'gw_bb'}),
    on='period', how='left'
)

# Calculate errors
merged['gcpi_error_new'] = merged['gcpi'] - merged['gcpi_new']
merged['gcpi_error_bb'] = merged['gcpi'] - merged['gcpi_bb']
merged['gw_error_new'] = merged['gw'] - merged['gw_new']
merged['gw_error_bb'] = merged['gw'] - merged['gw_bb']

# Filter to periods with both forecast and realized data
valid_gcpi = merged.dropna(subset=['gcpi', 'gcpi_new', 'gcpi_bb'])
valid_gw = merged.dropna(subset=['gw', 'gw_new', 'gw_bb'])

if len(valid_gcpi) > 0:
    print(f"\nInflation Forecast Errors ({len(valid_gcpi)} quarters):")
    print(f"  New Model:")
    print(f"    Mean Error (ME):  {valid_gcpi['gcpi_error_new'].mean():+.2f} pp")
    print(f"    Mean Abs Error:   {valid_gcpi['gcpi_error_new'].abs().mean():.2f} pp")
    print(f"    RMSE:             {np.sqrt((valid_gcpi['gcpi_error_new']**2).mean()):.2f} pp")
    print(f"  BB Model:")
    print(f"    Mean Error (ME):  {valid_gcpi['gcpi_error_bb'].mean():+.2f} pp")
    print(f"    Mean Abs Error:   {valid_gcpi['gcpi_error_bb'].abs().mean():.2f} pp")
    print(f"    RMSE:             {np.sqrt((valid_gcpi['gcpi_error_bb']**2).mean()):.2f} pp")

if len(valid_gw) > 0:
    print(f"\nWage Growth Forecast Errors ({len(valid_gw)} quarters):")
    print(f"  New Model:")
    print(f"    Mean Error (ME):  {valid_gw['gw_error_new'].mean():+.2f} pp")
    print(f"    Mean Abs Error:   {valid_gw['gw_error_new'].abs().mean():.2f} pp")
    print(f"    RMSE:             {np.sqrt((valid_gw['gw_error_new']**2).mean()):.2f} pp")
    print(f"  BB Model:")
    print(f"    Mean Error (ME):  {valid_gw['gw_error_bb'].mean():+.2f} pp")
    print(f"    Mean Abs Error:   {valid_gw['gw_error_bb'].abs().mean():.2f} pp")
    print(f"    RMSE:             {np.sqrt((valid_gw['gw_error_bb']**2).mean()):.2f} pp")

# Print quarter-by-quarter comparison
print("\n" + "-"*80)
print("QUARTER-BY-QUARTER COMPARISON")
print("-"*80)
print("\nInflation (gcpi):")
print(f"{'Quarter':<12} {'Realized':>10} {'New Model':>12} {'BB Model':>12} {'New Err':>10} {'BB Err':>10}")
print("-"*68)
for _, row in valid_gcpi.iterrows():
    q = period_to_quarter_label(row['period'])
    print(f"{q:<12} {row['gcpi']:>10.2f} {row['gcpi_new']:>12.2f} {row['gcpi_bb']:>12.2f} {row['gcpi_error_new']:>+10.2f} {row['gcpi_error_bb']:>+10.2f}")

print("\nWage Growth (gw):")
print(f"{'Quarter':<12} {'Realized':>10} {'New Model':>12} {'BB Model':>12} {'New Err':>10} {'BB Err':>10}")
print("-"*68)
for _, row in valid_gw.iterrows():
    q = period_to_quarter_label(row['period'])
    print(f"{q:<12} {row['gw']:>10.2f} {row['gw_new']:>12.2f} {row['gw_bb']:>12.2f} {row['gw_error_new']:>+10.2f} {row['gw_error_bb']:>+10.2f}")

# Save error analysis
merged.to_excel(output_dir / 'forecast_error_analysis.xlsx', index=False)
print(f"\n  Detailed error analysis saved to: {output_dir / 'forecast_error_analysis.xlsx'}")

# %%
print("\n" + "="*80)
print("FORECAST VS REALIZED COMPARISON COMPLETE!")
print("="*80)
print(f"\nOutput files saved to: {output_dir}")
print("  - figure_inflation_forecast_vs_realized.png/pdf")
print("  - figure_wage_forecast_vs_realized.png/pdf")
print("  - figure_forecast_vs_realized_combined.png/pdf")
print("  - forecast_error_analysis.xlsx")
print("\n" + "-"*80)
print("TO ADD MORE RECENT DATA:")
print("-"*80)
print("""
1. Download quarterly data from FRED:
   - CPIAUCSL: https://fred.stlouisfed.org/series/CPIAUCSL
   - ECIWAG:   https://fred.stlouisfed.org/series/ECIWAG

2. Edit realized_data_raw.csv and paste in the raw values:
   Date,CPIAUCSL,ECIWAG,VOVERU,...
   2023 Q3,305.691,162.8,1.52,...
   2023 Q4,307.234,164.1,1.45,...

3. Re-run this script - it will automatically calculate:
   - gcpi = 400 * ln(CPI_t / CPI_{t-1})
   - gw = 400 * ln(ECI_t / ECI_{t-1})

Note: The script uses the same formulas as regression_new_model_refactored.py
""")

# %%
