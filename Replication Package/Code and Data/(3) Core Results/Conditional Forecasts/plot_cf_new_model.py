# %%
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modified Bernanke-Blanchard Model - Conditional Forecast Plots
Liang & Sun (2025)

This file generates plots of the conditional forecasts from the new model.

Generates:
- Figure 14: Inflation projections for alternative paths of v/u
- Figure NEW: Shortage projections (endogenous)
- V/U paths
- Combined figures

Configuration flags at the top control which specification to plot.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

# %%
#****************************CONFIGURATION**************************************
# These flags MUST match the specification used in cond_forecast_new_model.py!

USE_PRE_COVID_SAMPLE = False       # Use pre-COVID sample estimates (typically False for forecasts)
USE_LOG_CU_WAGES = False           # True = log(CU), False = level CU in wage equation
USE_CONTEMP_CU = False             # True = CU lags 0-4, False = CU lags 1-4
USE_DETRENDED_EXCESS_DEMAND = True # Detrend excess demand in shortage equation

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

def get_spec_short_name():
    """Get a short name for the specification (for display)."""
    parts = []
    if USE_PRE_COVID_SAMPLE:
        parts.append("Pre-COVID")
    else:
        parts.append("Full Sample")
    if USE_LOG_CU_WAGES:
        parts.append("Log CU")
    else:
        parts.append("Level CU")
    if USE_CONTEMP_CU:
        parts.append("L0-L4")
    else:
        parts.append("L1-L4")
    if USE_DETRENDED_EXCESS_DEMAND:
        parts.append("Detrended ED")
    return ", ".join(parts)

SPEC_DIR_NAME = get_spec_dir_name()
SPEC_SHORT_NAME = get_spec_short_name()

# Input path - output from cond_forecast_new_model.py
input_dir = BASE_DIR / "(3) Core Results/Conditional Forecasts/Output Data Python (New Model)" / SPEC_DIR_NAME

# Output path for figures
output_dir = BASE_DIR / "(3) Core Results/Conditional Forecasts/Figures Python (New Model)" / SPEC_DIR_NAME
output_dir.mkdir(parents=True, exist_ok=True)

#*********************************************************************************

print("="*80)
print("CONDITIONAL FORECAST PLOTS - NEW MODEL")
print("Liang & Sun (2025)")
print("="*80)

print(f"\nSpecification: {SPEC_SHORT_NAME}")
print(f"  USE_PRE_COVID_SAMPLE:        {USE_PRE_COVID_SAMPLE}")
print(f"  USE_LOG_CU_WAGES:            {USE_LOG_CU_WAGES}")
print(f"  USE_CONTEMP_CU:              {USE_CONTEMP_CU}")
print(f"  USE_DETRENDED_EXCESS_DEMAND: {USE_DETRENDED_EXCESS_DEMAND}")

print(f"\nInput dir:  {input_dir}")
print(f"Output dir: {output_dir}")
print(f"Input exists: {input_dir.exists()}")

print("\nLoading conditional forecast results...")

# Load forecast data
terminal_low = pd.read_excel(input_dir / 'terminal_low.xlsx')
terminal_mid = pd.read_excel(input_dir / 'terminal_mid.xlsx')
terminal_high = pd.read_excel(input_dir / 'terminal_high.xlsx')

print(f"Loaded {len(terminal_low)} periods of forecast data")

# %%
# Convert period to datetime if needed
for df in [terminal_low, terminal_mid, terminal_high]:
    if not pd.api.types.is_datetime64_any_dtype(df['period']):
        df['period'] = pd.to_datetime(df['period'])

# Create combined DataFrame for plotting
cf_data = pd.DataFrame({
    'period': terminal_low['period'],
    'gcpi_low': terminal_low['gcpi_simul'],
    'gcpi_mid': terminal_mid['gcpi_simul'],
    'gcpi_high': terminal_high['gcpi_simul'],
    'shortage_low': terminal_low['shortage_simul'],
    'shortage_mid': terminal_mid['shortage_simul'],
    'shortage_high': terminal_high['shortage_simul'],
    'vu_low': terminal_low['vu_simul'],
    'vu_mid': terminal_mid['vu_simul'],
    'vu_high': terminal_high['vu_simul'],
    'gw_low': terminal_low['gw_simul'],
    'gw_mid': terminal_mid['gw_simul'],
    'gw_high': terminal_high['gw_simul']
})

# Filter to plotting range: 2022 Q4 to 2027 Q1
cf_data = cf_data[(cf_data['period'] >= '2022-10-01') & (cf_data['period'] <= '2027-01-01')].copy()
cf_data = cf_data.reset_index(drop=True)

print(f"Filtered to {len(cf_data)} periods for plotting")
print(f"Period range: {cf_data['period'].min()} to {cf_data['period'].max()}")

# %%
# Helper function to format quarter labels
def period_to_quarter_label(dt):
    """Convert datetime to 'YYYY QN' format"""
    if pd.isna(dt):
        return ''
    return f"{dt.year} Q{(dt.month - 1) // 3 + 1}"

# Create quarter labels
quarter_labels = [period_to_quarter_label(p) for p in cf_data['period']]

# Tick positions (every 2 quarters)
tick_positions = list(range(0, len(cf_data), 2))
tick_labels = [quarter_labels[i] for i in tick_positions]

# %%
# Colorblind-friendly palette
# Based on Wong (2011) "Points of view: Color blindness" Nature Methods
colors = {
    'low': '#0072B2',      # Blue (v/u -> 0.8)
    'mid': '#D55E00',      # Vermillion (v/u -> 1.2)
    'high': '#E69F00',     # Orange (v/u -> 1.8)
    'new_model': '#009E73', # Teal (for comparison plots)
    'bb_model': '#CC79A7',  # Pink (for comparison plots)
}

# Common plot styling
def setup_plot_style():
    """Set up common plot styling"""
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

setup_plot_style()

# %%
# Figure 14: INFLATION PROJECTIONS
print("\nCreating Figure 14: Conditional Inflation Forecasts (New Model)...")

fig, ax = plt.subplots(figsize=(12, 7))

ax.plot(cf_data['period'], cf_data['gcpi_low'], color=colors['low'], linewidth=2,
        label='v/u = 0.8')
ax.plot(cf_data['period'], cf_data['gcpi_mid'], color=colors['mid'], linewidth=2,
        label='v/u = 1.2 = (v/u)*')
ax.plot(cf_data['period'], cf_data['gcpi_high'], color=colors['high'], linewidth=2,
        label='v/u = 1.8')

ax.set_title('Inflation projections by labor market tightness',
             fontsize=17.5, fontweight='normal')
ax.set_xlabel('Quarter', fontsize=16)
ax.set_ylabel('Percent', fontsize=16)

ax.set_ylim(1.5, 5.0)
ax.set_yticks(np.arange(1.5, 5.5, 0.5))

ax.set_xticks([cf_data['period'].iloc[i] for i in tick_positions])
ax.set_xticklabels(tick_labels, rotation=0, ha='right')

ax.yaxis.grid(True, linestyle='-', linewidth=0.5, color='lightgray')
ax.xaxis.grid(False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.13), ncol=3,
          frameon=True, edgecolor='black', fancybox=False)

plt.tight_layout()
plt.savefig(output_dir / 'figure_14_inflation_new_model.png', dpi=300, bbox_inches='tight')
plt.savefig(output_dir / 'figure_14_inflation_new_model.pdf', bbox_inches='tight')
print(f"  Saved to {output_dir / 'figure_14_inflation_new_model.png'}")
plt.show()

# %%
# NEW FIGURE: SHORTAGE PROJECTIONS (Endogenous)
print("\nCreating Shortage Projections (New - Endogenous)...")

fig, ax = plt.subplots(figsize=(12, 7))

ax.plot(cf_data['period'], cf_data['shortage_low'], color=colors['low'], linewidth=2,
        label='v/u = 0.8')
ax.plot(cf_data['period'], cf_data['shortage_mid'], color=colors['mid'], linewidth=2,
        label='v/u = 1.2 = (v/u)*')
ax.plot(cf_data['period'], cf_data['shortage_high'], color=colors['high'], linewidth=2,
        label='v/u = 1.8')

ax.set_title('Shortage Index Projections (Endogenous in New Model)',
             fontsize=17.5, fontweight='normal')
ax.set_xlabel('Quarter', fontsize=16)
ax.set_ylabel('Shortage Index', fontsize=16)

ax.set_xticks([cf_data['period'].iloc[i] for i in tick_positions])
ax.set_xticklabels(tick_labels, rotation=45, ha='right')

ax.yaxis.grid(True, linestyle='-', linewidth=0.5, color='lightgray')
ax.xaxis.grid(False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3,
          frameon=True, edgecolor='black', fancybox=False)

plt.tight_layout()
plt.savefig(output_dir / 'figure_shortage_projection.png', dpi=300, bbox_inches='tight')
plt.savefig(output_dir / 'figure_shortage_projection.pdf', bbox_inches='tight')
print(f"  Saved to {output_dir / 'figure_shortage_projection.png'}")
plt.show()

# %%
# V/U PATHS
print("\nCreating V/U paths figure...")

fig, ax = plt.subplots(figsize=(12, 7))

ax.plot(cf_data['period'], cf_data['vu_low'], color=colors['low'], linewidth=2,
        label='v/u -> 0.8')
ax.plot(cf_data['period'], cf_data['vu_mid'], color=colors['mid'], linewidth=2,
        label='v/u -> 1.2')
ax.plot(cf_data['period'], cf_data['vu_high'], color=colors['high'], linewidth=2,
        label='v/u -> 1.8')

ax.set_title('V/U Ratio Paths for Conditional Forecasts',
             fontsize=17.5, fontweight='normal')
ax.set_xlabel('Quarter', fontsize=16)
ax.set_ylabel('V/U Ratio', fontsize=16)

ax.set_xticks([cf_data['period'].iloc[i] for i in tick_positions])
ax.set_xticklabels(tick_labels, rotation=45, ha='right')

ax.yaxis.grid(True, linestyle='-', linewidth=0.5, color='lightgray')
ax.xaxis.grid(False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3,
          frameon=True, edgecolor='black', fancybox=False)

plt.tight_layout()
plt.savefig(output_dir / 'figure_vu_paths.png', dpi=300, bbox_inches='tight')
plt.savefig(output_dir / 'figure_vu_paths.pdf', bbox_inches='tight')
print(f"  Saved to {output_dir / 'figure_vu_paths.png'}")
plt.show()

# %%
# WAGE GROWTH PROJECTIONS
print("\nCreating Wage Growth projections...")

fig, ax = plt.subplots(figsize=(12, 7))

ax.plot(cf_data['period'], cf_data['gw_low'], color=colors['low'], linewidth=2,
        label='v/u = 0.8')
ax.plot(cf_data['period'], cf_data['gw_mid'], color=colors['mid'], linewidth=2,
        label='v/u = 1.2 = (v/u)*')
ax.plot(cf_data['period'], cf_data['gw_high'], color=colors['high'], linewidth=2,
        label='v/u = 1.8')

ax.set_title('Wage growth projections by labor market tightness',
             fontsize=17.5, fontweight='normal')
ax.set_xlabel('Quarter', fontsize=16)
ax.set_ylabel('Percent', fontsize=16)

ax.set_xticks([cf_data['period'].iloc[i] for i in tick_positions])
ax.set_xticklabels(tick_labels, rotation=0, ha='right')

ax.yaxis.grid(True, linestyle='-', linewidth=0.5, color='lightgray')
ax.xaxis.grid(False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.13), ncol=3,
          frameon=True, edgecolor='black', fancybox=False)

plt.tight_layout()
plt.savefig(output_dir / 'figure_wage_projection.png', dpi=300, bbox_inches='tight')
plt.savefig(output_dir / 'figure_wage_projection.pdf', bbox_inches='tight')
print(f"  Saved to {output_dir / 'figure_wage_projection.png'}")
plt.show()

# %%
# COMBINED 4-PANEL FIGURE
print("\nCreating combined 4-panel figure...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Panel 1: V/U paths
ax1 = axes[0, 0]
ax1.plot(cf_data['period'], cf_data['vu_low'], color=colors['low'], linewidth=2, label='v/u -> 0.8')
ax1.plot(cf_data['period'], cf_data['vu_mid'], color=colors['mid'], linewidth=2, label='v/u -> 1.2')
ax1.plot(cf_data['period'], cf_data['vu_high'], color=colors['high'], linewidth=2, label='v/u -> 1.8')
ax1.set_title('A. V/U Ratio Paths', fontsize=14)
ax1.set_ylabel('V/U Ratio', fontsize=12)
ax1.set_xticks([cf_data['period'].iloc[i] for i in tick_positions])
ax1.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=10)
ax1.yaxis.grid(True, linestyle='-', linewidth=0.5, color='lightgray')
ax1.xaxis.grid(False)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.legend(loc='best', fontsize=10, frameon=True, edgecolor='black')

# Panel 2: Shortage (endogenous)
ax2 = axes[0, 1]
ax2.plot(cf_data['period'], cf_data['shortage_low'], color=colors['low'], linewidth=2, label='v/u = 0.8')
ax2.plot(cf_data['period'], cf_data['shortage_mid'], color=colors['mid'], linewidth=2, label='v/u = 1.2')
ax2.plot(cf_data['period'], cf_data['shortage_high'], color=colors['high'], linewidth=2, label='v/u = 1.8')
ax2.set_title('B. Shortage Index (Endogenous)', fontsize=14)
ax2.set_ylabel('Index', fontsize=12)
ax2.set_xticks([cf_data['period'].iloc[i] for i in tick_positions])
ax2.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=10)
ax2.yaxis.grid(True, linestyle='-', linewidth=0.5, color='lightgray')
ax2.xaxis.grid(False)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.legend(loc='best', fontsize=10, frameon=True, edgecolor='black')

# Panel 3: Wage growth
ax3 = axes[1, 0]
ax3.plot(cf_data['period'], cf_data['gw_low'], color=colors['low'], linewidth=2, label='v/u = 0.8')
ax3.plot(cf_data['period'], cf_data['gw_mid'], color=colors['mid'], linewidth=2, label='v/u = 1.2')
ax3.plot(cf_data['period'], cf_data['gw_high'], color=colors['high'], linewidth=2, label='v/u = 1.8')
ax3.set_title('C. Wage Growth', fontsize=14)
ax3.set_ylabel('Percent', fontsize=12)
ax3.set_xticks([cf_data['period'].iloc[i] for i in tick_positions])
ax3.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=10)
ax3.yaxis.grid(True, linestyle='-', linewidth=0.5, color='lightgray')
ax3.xaxis.grid(False)
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)
ax3.legend(loc='best', fontsize=10, frameon=True, edgecolor='black')

# Panel 4: Inflation
ax4 = axes[1, 1]
ax4.plot(cf_data['period'], cf_data['gcpi_low'], color=colors['low'], linewidth=2, label='v/u = 0.8')
ax4.plot(cf_data['period'], cf_data['gcpi_mid'], color=colors['mid'], linewidth=2, label='v/u = 1.2')
ax4.plot(cf_data['period'], cf_data['gcpi_high'], color=colors['high'], linewidth=2, label='v/u = 1.8')
ax4.set_title('D. Inflation (CPI)', fontsize=14)
ax4.set_ylabel('Percent', fontsize=12)
ax4.set_ylim(1.5, 4.0)
ax4.set_yticks(np.arange(1.5, 4.5, 0.5))
ax4.set_xticks([cf_data['period'].iloc[i] for i in tick_positions])
ax4.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=10)
ax4.yaxis.grid(True, linestyle='-', linewidth=0.5, color='lightgray')
ax4.xaxis.grid(False)
ax4.spines['top'].set_visible(False)
ax4.spines['right'].set_visible(False)
ax4.legend(loc='best', fontsize=10, frameon=True, edgecolor='black')

plt.suptitle('Conditional Forecasts: Modified Bernanke-Blanchard Model (Liang & Sun 2025)',
             fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(output_dir / 'figure_combined_4panel.png', dpi=300, bbox_inches='tight')
plt.savefig(output_dir / 'figure_combined_4panel.pdf', bbox_inches='tight')
print(f"  Saved to {output_dir / 'figure_combined_4panel.png'}")
plt.show()


# %%
# COMPARISON WITH BB MODEL (if available)
print("\n" + "="*80)
print("COMPARISON WITH ORIGINAL BB MODEL")
print("="*80)

bb_input_dir = Path("/Users/johnathansun/Documents/ec1499/Replication Package/Code and Data/(3) Core Results/Conditional Forecasts/Output Data Python")

if (bb_input_dir / 'terminal_mid.xlsx').exists():
    print("\nLoading original BB model forecasts for comparison...")

    bb_low = pd.read_excel(bb_input_dir / 'terminal_low.xlsx')
    bb_mid = pd.read_excel(bb_input_dir / 'terminal_mid.xlsx')
    bb_high = pd.read_excel(bb_input_dir / 'terminal_high.xlsx')

    for df in [bb_low, bb_mid, bb_high]:
        if not pd.api.types.is_datetime64_any_dtype(df['period']):
            df['period'] = pd.to_datetime(df['period'])

    # Filter to same range
    bb_low = bb_low[(bb_low['period'] >= '2022-10-01') & (bb_low['period'] <= '2027-01-01')].reset_index(drop=True)
    bb_mid = bb_mid[(bb_mid['period'] >= '2022-10-01') & (bb_mid['period'] <= '2027-01-01')].reset_index(drop=True)
    bb_high = bb_high[(bb_high['period'] >= '2022-10-01') & (bb_high['period'] <= '2027-01-01')].reset_index(drop=True)

    # Create comparison figure for v/u -> 1.2 scenario
    print("\nCreating BB vs New Model comparison (v/u -> 1.2)...")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Left: Inflation comparison
    ax1 = axes[0]
    ax1.plot(cf_data['period'], cf_data['gcpi_mid'], color=colors['new_model'], linewidth=2, label='New Model')
    ax1.plot(bb_mid['period'], bb_mid['gcpi_simul'], color=colors['bb_model'], linewidth=2, linestyle='--', label='BB Model')
    ax1.set_title('Inflation Forecast: BB vs New Model (v/u -> 1.2)', fontsize=14)
    ax1.set_ylabel('Percent', fontsize=12)
    ax1.set_ylim(1.5, 4.0)
    ax1.set_xticks([cf_data['period'].iloc[i] for i in tick_positions])
    ax1.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=10)
    ax1.yaxis.grid(True, linestyle='-', linewidth=0.5, color='lightgray')
    ax1.xaxis.grid(False)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.legend(loc='best', fontsize=12, frameon=True, edgecolor='black')

    # Right: Shortage comparison (BB is exogenous/constant, New is endogenous)
    ax2 = axes[1]
    ax2.plot(cf_data['period'], cf_data['shortage_mid'], color=colors['new_model'], linewidth=2, label='New Model (endogenous)')
    ax2.plot(bb_mid['period'], bb_mid['shortage_simul'], color=colors['bb_model'], linewidth=2, linestyle='--', label='BB Model (exogenous)')
    ax2.set_title('Shortage: BB (exogenous) vs New Model (endogenous)', fontsize=14)
    ax2.set_ylabel('Index', fontsize=12)
    ax2.set_xticks([cf_data['period'].iloc[i] for i in tick_positions])
    ax2.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=10)
    ax2.yaxis.grid(True, linestyle='-', linewidth=0.5, color='lightgray')
    ax2.xaxis.grid(False)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.legend(loc='best', fontsize=12, frameon=True, edgecolor='black')

    plt.suptitle('Comparison: Original BB Model vs Modified Model',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / 'figure_comparison_bb_vs_new.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'figure_comparison_bb_vs_new.pdf', bbox_inches='tight')
    print(f"  Saved to {output_dir / 'figure_comparison_bb_vs_new.png'}")
    plt.show()

    # STANDALONE INFLATION COMPARISON FIGURE
    print("\nCreating standalone inflation comparison figure...")

    fig, ax = plt.subplots(figsize=(12, 7))

    ax.plot(cf_data['period'], cf_data['gcpi_mid'], color=colors['new_model'], linewidth=2, label='New Model')
    ax.plot(bb_mid['period'], bb_mid['gcpi_simul'], color=colors['bb_model'], linewidth=2, linestyle='--', label='BB Model')

    ax.set_title('Comparison of inflation projections (v/u = 1.2)',
                 fontsize=17.5, fontweight='normal')
    ax.set_xlabel('Quarter', fontsize=16)
    ax.set_ylabel('Percent', fontsize=16)

    ax.set_ylim(1.5, 5.0)
    ax.set_yticks(np.arange(1.5, 5.5, 0.5))

    ax.set_xticks([cf_data['period'].iloc[i] for i in tick_positions])
    ax.set_xticklabels(tick_labels, rotation=0, ha='right')

    ax.yaxis.grid(True, linestyle='-', linewidth=0.5, color='lightgray')
    ax.xaxis.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=2,
              frameon=True, edgecolor='black', fancybox=False)

    plt.tight_layout()
    plt.savefig(output_dir / 'figure_inflation_comparison_only.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'figure_inflation_comparison_only.pdf', bbox_inches='tight')
    print(f"  Saved to {output_dir / 'figure_inflation_comparison_only.png'}")
    plt.show()

    # Print comparison statistics
    print("\n--- Model Comparison (v/u -> 1.2 scenario) ---")
    print(f"\nTerminal inflation:")
    print(f"  BB Model:  {bb_mid['gcpi_simul'].iloc[-1]:.2f}%")
    print(f"  New Model: {cf_data['gcpi_mid'].iloc[-1]:.2f}%")
    print(f"  Difference: {cf_data['gcpi_mid'].iloc[-1] - bb_mid['gcpi_simul'].iloc[-1]:.2f} pp")

    print(f"\nTerminal shortage:")
    print(f"  BB Model (exogenous):  {bb_mid['shortage_simul'].iloc[-1]:.2f}")
    print(f"  New Model (endogenous): {cf_data['shortage_mid'].iloc[-1]:.2f}")

else:
    print("\nOriginal BB model forecasts not found. Run cond_forecast.py first.")


# %%
# Print summary statistics
print("\n" + "="*80)
print("CONDITIONAL FORECAST SUMMARY (New Model)")
print("="*80)

print(f"\nInflation forecasts by scenario:")
print(f"\n  Scenario: v/u -> 0.8")
print(f"    Start (2022 Q4): {cf_data['gcpi_low'].iloc[0]:.2f}%")
print(f"    End (2027 Q1):   {cf_data['gcpi_low'].iloc[-1]:.2f}%")

print(f"\n  Scenario: v/u -> 1.2 (target)")
print(f"    Start (2022 Q4): {cf_data['gcpi_mid'].iloc[0]:.2f}%")
print(f"    End (2027 Q1):   {cf_data['gcpi_mid'].iloc[-1]:.2f}%")

print(f"\n  Scenario: v/u -> 1.8")
print(f"    Start (2022 Q4): {cf_data['gcpi_high'].iloc[0]:.2f}%")
print(f"    End (2027 Q1):   {cf_data['gcpi_high'].iloc[-1]:.2f}%")

print(f"\nShortage forecasts (endogenous):")
print(f"\n  Scenario: v/u -> 0.8")
print(f"    Start: {cf_data['shortage_low'].iloc[0]:.2f}")
print(f"    End:   {cf_data['shortage_low'].iloc[-1]:.2f}")

print(f"\n  Scenario: v/u -> 1.2")
print(f"    Start: {cf_data['shortage_mid'].iloc[0]:.2f}")
print(f"    End:   {cf_data['shortage_mid'].iloc[-1]:.2f}")

print(f"\n  Scenario: v/u -> 1.8")
print(f"    Start: {cf_data['shortage_high'].iloc[0]:.2f}")
print(f"    End:   {cf_data['shortage_high'].iloc[-1]:.2f}")

print("\n" + "="*80)
print("PLOTTING COMPLETE!")
print("="*80)
print(f"\nOutput files saved to: {output_dir}")
print("  - figure_14_inflation_new_model.png/pdf")
print("  - figure_shortage_projection.png/pdf")
print("  - figure_vu_paths.png/pdf")
print("  - figure_wage_projection.png/pdf")
print("  - figure_combined_4panel.png/pdf")
print("  - figure_comparison_bb_vs_new.png/pdf (if BB data available)")
print("  - figure_inflation_comparison_only.png/pdf (if BB data available)")
print("\n")

# %%
