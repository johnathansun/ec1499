# %%
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modified Bernanke-Blanchard Model - IRF Plots
Liang & Sun (2025)

This file generates IRF plots for the modified model with support for
multiple model specifications.

Configuration flags at the top control which specification to plot.
The script automatically loads the appropriate IRF results based on
the configuration.

Generates:
- Figure 10: Responses to energy, food, and shortage shocks
- Figure 11: Response to V/U shock
- Figure 12: Response to GSCPI shock (via endogenous shortage)
- Figure 13: Response to capacity utilization shock (via wages -> ED -> shortage)
- Figure 14: Response to potential GDP shock (via ED -> shortage)
- Figure 15: Persistent vs one-time shocks comparison
- Figure 16: All shocks combined
- Figure 17: Endogenous feedback loop (V/U shock)
- Figure 18: Endogenous feedback loop (Capacity utilization shock)
- Figure 19: Wage response to labor market shocks
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# %%
#****************************CONFIGURATION**************************************
# These flags MUST match the specification used in irf_simulations_new_model.py

USE_PRE_COVID_SAMPLE = False       # Use pre-COVID sample estimates
USE_LOG_CU_WAGES = False          # True = log(CU), False = level CU
USE_CONTEMP_CU = False             # True = CU lags 0-4, False = CU lags 1-4
USE_DETRENDED_EXCESS_DEMAND = True  # Detrend excess demand in shortage eq

#****************************PATH CONFIGURATION*********************************

BASE_DIR = Path("/Users/johnathansun/Documents/ec1499/Replication Package/Code and Data")

# Build directory names based on configuration
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
    """Get a short name for the specification (for plot titles)."""
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
    return ", ".join(parts)

SPEC_DIR_NAME = get_spec_dir_name()
SPEC_SHORT_NAME = get_spec_short_name()

# Input location - output from irf_simulations_new_model.py
input_dir = BASE_DIR / "(3) Core Results/IRFs/Output Data Python (New Model)" / SPEC_DIR_NAME

# Output location for figures
output_dir = BASE_DIR / "(3) Core Results/IRFs/Figures Python (New Model)" / SPEC_DIR_NAME
output_dir.mkdir(parents=True, exist_ok=True)

#*******************************************************************************

def print_configuration():
    """Print current configuration for verification."""
    print("=" * 80)
    print("MODIFIED MODEL IRF PLOTS")
    print("Liang & Sun (2025)")
    print("=" * 80)
    print("\nCONFIGURATION:")
    print(f"  USE_PRE_COVID_SAMPLE:        {USE_PRE_COVID_SAMPLE}")
    print(f"  USE_LOG_CU_WAGES:            {USE_LOG_CU_WAGES}")
    print(f"  USE_CONTEMP_CU:              {USE_CONTEMP_CU}")
    print(f"  USE_DETRENDED_EXCESS_DEMAND: {USE_DETRENDED_EXCESS_DEMAND}")
    print(f"\nSpecification: {SPEC_SHORT_NAME}")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")

print_configuration()

# %%
# VERIFY PATHS AND LOAD DATA
print("\nLoading IRF results...")

if not input_dir.exists():
    raise FileNotFoundError(
        f"IRF results directory not found: {input_dir}\n"
        f"Run irf_simulations_new_model.py first with matching configuration."
    )

# Load IRF data - original BB shocks
irf_energy = pd.read_excel(input_dir / 'results_energy.xlsx')
irf_food = pd.read_excel(input_dir / 'results_food.xlsx')
irf_shortage = pd.read_excel(input_dir / 'results_shortage.xlsx')
irf_vu = pd.read_excel(input_dir / 'results_vu.xlsx')

# Load IRF data - new model shocks
irf_gscpi = pd.read_excel(input_dir / 'results_gscpi.xlsx')
irf_gcu = pd.read_excel(input_dir / 'results_gcu.xlsx')
irf_ngdppot = pd.read_excel(input_dir / 'results_ngdppot.xlsx')

# Load IRF data - persistent shocks
irf_gscpi_persistent = pd.read_excel(input_dir / 'results_gscpi_persistent.xlsx')
irf_gcu_persistent = pd.read_excel(input_dir / 'results_gcu_persistent.xlsx')
irf_ngdppot_persistent = pd.read_excel(input_dir / 'results_ngdppot_persistent.xlsx')

# Load configuration to verify shock magnitudes
try:
    irf_config = pd.read_excel(input_dir / 'irf_config.xlsx')
    print("\nConfiguration loaded from irf_config.xlsx:")
    for _, row in irf_config.iterrows():
        print(f"  {row['Parameter']}: {row['Value']}")
except FileNotFoundError:
    print("\nWarning: irf_config.xlsx not found - cannot verify shock magnitudes")
    irf_config = None

print(f"\nLoaded {len(irf_energy)} periods of IRF data")

# %%
# Filter to periods 5-20 (quarters 1-16 after shock)
def filter_irf(df):
    return df[(df['period'] >= 5) & (df['period'] <= 20)].copy()

irf_energy_f = filter_irf(irf_energy)
irf_food_f = filter_irf(irf_food)
irf_shortage_f = filter_irf(irf_shortage)
irf_vu_f = filter_irf(irf_vu)
irf_gscpi_f = filter_irf(irf_gscpi)
irf_gcu_f = filter_irf(irf_gcu)
irf_ngdppot_f = filter_irf(irf_ngdppot)
irf_gscpi_persistent_f = filter_irf(irf_gscpi_persistent)
irf_gcu_persistent_f = filter_irf(irf_gcu_persistent)
irf_ngdppot_persistent_f = filter_irf(irf_ngdppot_persistent)

quarters = np.arange(1, len(irf_energy_f) + 1)

print(f"Plotting {len(quarters)} quarters of IRF responses")

# %%
# Colorblind-friendly palette (matching decomposition plots)
colors = {
    'Energy': '#0072B2',        # Blue
    'Food': '#56B4E9',          # Sky blue
    'Shortage': '#F0E442',      # Yellow
    'V/U': '#D55E00',           # Vermillion
    'Excess Demand': '#882255', # Dark magenta/wine
    'GSCPI': '#009E73',         # Teal
    'Capacity Util': '#CC79A7', # Pink
    'Potential GDP': '#117733', # Dark green
    'Wage Level': '#E69F00',    # Orange
    'Persistent': '#000000',    # Black (for persistent versions)
    'One-time': '#888888'       # Grey (for one-time versions)
}

# %%
# Common plot styling
def setup_plot_style():
    """Set up common plot styling"""
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 16,
        'axes.titlesize': 17.5,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 11,
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
# =============================================================================
# FIGURE 10: ORIGINAL BB SUPPLY SHOCKS
# =============================================================================
print("\nCreating Figure 10: Original BB Supply Shocks...")

fig, ax = plt.subplots(figsize=(12, 7))

ax.plot(quarters, irf_energy_f['gcpi_simul'].values, color=colors['Energy'],
        linewidth=2, label='Energy Prices')
ax.plot(quarters, irf_food_f['gcpi_simul'].values, color=colors['Food'],
        linewidth=2, label='Food Prices')

ax.set_title(f'Inflation response to food/energy shocks',
             fontsize=17.5, fontweight='normal')
ax.set_xlabel('Quarter', fontsize=16)
ax.set_ylabel('Percent', fontsize=16)

ax.set_xlim(0.5, 16.5)
ax.set_xticks(range(1, 17))
ax.set_ylim(-0.5, 3)

ax.yaxis.grid(True, linestyle='-', linewidth=0.5, color='lightgray')
ax.xaxis.grid(False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.axhline(y=0, color='black', linewidth=0.5)

ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.10), ncol=3,
          frameon=True, edgecolor='black', fancybox=False)

plt.tight_layout()
plt.savefig(output_dir / 'figure_10_irf_supply_shocks.png', dpi=300, bbox_inches='tight')
plt.savefig(output_dir / 'figure_10_irf_supply_shocks.pdf', bbox_inches='tight')
print(f"  Saved to {output_dir / 'figure_10_irf_supply_shocks.png'}")
plt.show()


# %%
print("\nCreating Figure 11: Original BB Supply Shocks...")

fig, ax = plt.subplots(figsize=(12, 7))

ax.plot(quarters, irf_gscpi_f['gcpi_simul'].values, color=colors['GSCPI'],
        linewidth=2, label='Supply Chain Pressure')
ax.plot(quarters, irf_gcu_f['gcpi_simul'].values, color=colors['Capacity Util'],
        linewidth=2, label='Capacity Utilization')
ax.plot(quarters, irf_ngdppot_f['gcpi_simul'].values, color=colors['Potential GDP'],
        linewidth=2, label='Potential GDP')

ax.set_title(f'Inflation response to additional shocks',
             fontsize=17.5, fontweight='normal')
ax.set_xlabel('Quarter', fontsize=16)
ax.set_ylabel('Percent', fontsize=16)

ax.set_xlim(0.5, 16.5)
ax.set_xticks(range(1, 17))
ax.set_ylim(-0.5, 0.75)

ax.yaxis.grid(True, linestyle='-', linewidth=0.5, color='lightgray')
ax.xaxis.grid(False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.axhline(y=0, color='black', linewidth=0.5)

ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.10), ncol=3,
          frameon=True, edgecolor='black', fancybox=False)

plt.tight_layout()
plt.savefig(output_dir / 'figure_11_supply_shock.pdf', bbox_inches='tight')


# %%
# =============================================================================
# FIGURE 11: V/U SHOCK
# =============================================================================
print("\nCreating Figure 11: V/U Shock...")

fig, ax = plt.subplots(figsize=(12, 7))

ax.plot(quarters, irf_vu_f['gcpi_simul'].values, color=colors['V/U'], linewidth=2)

ax.set_title(f'Inflation response to persistent V/U shock',
             fontsize=17.5, fontweight='normal')
ax.set_xlabel('Quarter', fontsize=16)
ax.set_ylabel('Percent', fontsize=16)

ax.set_xlim(0.5, 16.5)
ax.set_xticks(range(1, 17))
ax.set_ylim(0.0, 1.5)

ax.yaxis.grid(True, linestyle='-', linewidth=0.5, color='lightgray')
ax.xaxis.grid(False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(output_dir / 'figure_11_irf_vu_shock.png', dpi=300, bbox_inches='tight')
plt.savefig(output_dir / 'figure_11_irf_vu_shock.pdf', bbox_inches='tight')
print(f"  Saved to {output_dir / 'figure_11_irf_vu_shock.png'}")
plt.show()


# %%
# =============================================================================
# FIGURE 12 (NEW): GSCPI SHOCK (SUPPLY CHAIN PRESSURE)
# =============================================================================
print("\nCreating Figure 12: GSCPI Shock...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Panel A: Inflation response
ax1 = axes[0]
ax1.plot(quarters, irf_gscpi_f['gcpi_simul'].values, color=colors['GSCPI'],
        linewidth=2, label='One-time')
ax1.plot(quarters, irf_gscpi_persistent_f['gcpi_simul'].values, color=colors['GSCPI'],
        linewidth=2, linestyle='--', label='Persistent')

ax1.set_title('(A) Inflation Response', fontsize=14)
ax1.set_xlabel('Quarter', fontsize=12)
ax1.set_ylabel('Percent', fontsize=12)
ax1.set_xlim(0.5, 16.5)
ax1.set_xticks(range(1, 17, 2))
ax1.yaxis.grid(True, linestyle='-', linewidth=0.5, color='lightgray')
ax1.xaxis.grid(False)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.axhline(y=0, color='black', linewidth=0.5)
ax1.legend(loc='best', fontsize=10)

# Panel B: Shortage response
ax2 = axes[1]
ax2.plot(quarters, irf_gscpi_f['shortage_simul'].values, color=colors['GSCPI'],
        linewidth=2, label='One-time')
ax2.plot(quarters, irf_gscpi_persistent_f['shortage_simul'].values, color=colors['GSCPI'],
        linewidth=2, linestyle='--', label='Persistent')

ax2.set_title('(B) Shortage Response', fontsize=14)
ax2.set_xlabel('Quarter', fontsize=12)
ax2.set_ylabel('Shortage Index', fontsize=12)
ax2.set_xlim(0.5, 16.5)
ax2.set_xticks(range(1, 17, 2))
ax2.yaxis.grid(True, linestyle='-', linewidth=0.5, color='lightgray')
ax2.xaxis.grid(False)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.axhline(y=0, color='black', linewidth=0.5)
ax2.legend(loc='best', fontsize=10)

plt.suptitle(f'Response to supply chain pressure',
             fontsize=16)
plt.tight_layout()
plt.savefig(output_dir / 'figure_12_irf_gscpi_shock.png', dpi=300, bbox_inches='tight')
plt.savefig(output_dir / 'figure_12_irf_gscpi_shock.pdf', bbox_inches='tight')
print(f"  Saved to {output_dir / 'figure_12_irf_gscpi_shock.png'}")
plt.show()


# %%
# =============================================================================
# FIGURE 13 (NEW): CAPACITY UTILIZATION SHOCK
# =============================================================================
print("\nCreating Figure 13: Capacity Utilization Shock...")

fig, ax = plt.subplots(figsize=(12, 7))

ax.plot(quarters, irf_gcu_f['gcpi_simul'].values, color=colors['Capacity Util'],
        linewidth=2, label='One-time')
ax.plot(quarters, irf_gcu_persistent_f['gcpi_simul'].values, color=colors['Capacity Util'],
        linewidth=2, linestyle='--', label='Persistent')

cu_type = "log(CU)" if USE_LOG_CU_WAGES else "level CU"
ax.set_title(f'Inflation response to capacity utilization shock',
             fontsize=16, fontweight='normal')
ax.set_xlabel('Quarter', fontsize=16)
ax.set_ylabel('Percent', fontsize=16)

ax.set_xlim(0.5, 16.5)
ax.set_xticks(range(1, 17))

ax.yaxis.grid(True, linestyle='-', linewidth=0.5, color='lightgray')
ax.xaxis.grid(False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.axhline(y=0, color='black', linewidth=0.5)

ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.10), ncol=2,
          frameon=True, edgecolor='black', fancybox=False)

plt.tight_layout()
plt.savefig(output_dir / 'figure_13_irf_capacity_util.png', dpi=300, bbox_inches='tight')
plt.savefig(output_dir / 'figure_13_irf_capacity_util.pdf', bbox_inches='tight')
print(f"  Saved to {output_dir / 'figure_13_irf_capacity_util.png'}")
plt.show()


# %%
# =============================================================================
# FIGURE 14 (NEW): POTENTIAL GDP SHOCK
# =============================================================================
print("\nCreating Figure 14: Potential GDP Shock...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Panel A: Inflation response (note: should be negative for positive NGDPPOT shock)
ax1 = axes[0]
ax1.plot(quarters, irf_ngdppot_f['gcpi_simul'].values, color=colors['Potential GDP'],
        linewidth=2, label='One-time')
ax1.plot(quarters, irf_ngdppot_persistent_f['gcpi_simul'].values, color=colors['Potential GDP'],
        linewidth=2, linestyle='--', label='Persistent (rho=0.9)')

ax1.set_title('(A) Inflation Response', fontsize=14, fontweight='bold')
ax1.set_xlabel('Quarter', fontsize=12)
ax1.set_ylabel('Percent', fontsize=12)
ax1.set_xlim(0.5, 16.5)
ax1.set_xticks(range(1, 17, 2))
ax1.yaxis.grid(True, linestyle='-', linewidth=0.5, color='lightgray')
ax1.xaxis.grid(False)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.axhline(y=0, color='black', linewidth=0.5)
ax1.legend(loc='best', fontsize=10)

# Panel B: Excess Demand response (note: should be negative for positive NGDPPOT shock)
ax2 = axes[1]
ax2.plot(quarters, irf_ngdppot_f['ed_simul'].values, color=colors['Potential GDP'],
        linewidth=2, label='One-time')
ax2.plot(quarters, irf_ngdppot_persistent_f['ed_simul'].values, color=colors['Potential GDP'],
        linewidth=2, linestyle='--', label='Persistent (rho=0.9)')

ax2.set_title('(B) Excess Demand Response', fontsize=14, fontweight='bold')
ax2.set_xlabel('Quarter', fontsize=12)
ax2.set_ylabel('ED Index', fontsize=12)
ax2.set_xlim(0.5, 16.5)
ax2.set_xticks(range(1, 17, 2))
ax2.yaxis.grid(True, linestyle='-', linewidth=0.5, color='lightgray')
ax2.xaxis.grid(False)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.axhline(y=0, color='black', linewidth=0.5)
ax2.legend(loc='best', fontsize=10)

plt.suptitle(f'Response to Potential GDP (NGDPPOT) Shock\n(Higher potential GDP → lower excess demand → lower inflation)\n({SPEC_SHORT_NAME})',
             fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(output_dir / 'figure_14_irf_ngdppot_shock.png', dpi=300, bbox_inches='tight')
plt.savefig(output_dir / 'figure_14_irf_ngdppot_shock.pdf', bbox_inches='tight')
print(f"  Saved to {output_dir / 'figure_14_irf_ngdppot_shock.png'}")
plt.show()


# %%
# =============================================================================
# FIGURE 15 (NEW): PERSISTENT VS ONE-TIME SHOCKS COMPARISON
# =============================================================================
print("\nCreating Figure 15: Persistent vs One-Time Shocks...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Panel A: GSCPI
ax1 = axes[0]
ax1.plot(quarters, irf_gscpi_f['gcpi_simul'].values,
         color=colors['GSCPI'], linewidth=2, label='One-time')
ax1.plot(quarters, irf_gscpi_persistent_f['gcpi_simul'].values,
         color=colors['GSCPI'], linewidth=2, linestyle='--', label='Persistent (rho=0.9)')
ax1.set_title('(A) GSCPI Shock', fontsize=14, fontweight='bold')
ax1.set_xlabel('Quarter', fontsize=12)
ax1.set_ylabel('Inflation (percent)', fontsize=12)
ax1.set_xlim(0.5, 16.5)
ax1.set_xticks(range(1, 17, 2))
ax1.yaxis.grid(True, linestyle='-', linewidth=0.5, color='lightgray')
ax1.xaxis.grid(False)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.axhline(y=0, color='black', linewidth=0.5)
ax1.legend(loc='best', fontsize=10)

# Panel B: Capacity Utilization
ax2 = axes[1]
ax2.plot(quarters, irf_gcu_f['gcpi_simul'].values,
         color=colors['Capacity Util'], linewidth=2, label='One-time')
ax2.plot(quarters, irf_gcu_persistent_f['gcpi_simul'].values,
         color=colors['Capacity Util'], linewidth=2, linestyle='--', label='Persistent (rho=0.9)')
ax2.set_title('(B) Capacity Utilization Shock', fontsize=14, fontweight='bold')
ax2.set_xlabel('Quarter', fontsize=12)
ax2.set_ylabel('Inflation (percent)', fontsize=12)
ax2.set_xlim(0.5, 16.5)
ax2.set_xticks(range(1, 17, 2))
ax2.yaxis.grid(True, linestyle='-', linewidth=0.5, color='lightgray')
ax2.xaxis.grid(False)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.axhline(y=0, color='black', linewidth=0.5)
ax2.legend(loc='best', fontsize=10)

plt.suptitle(f'Inflation Response: One-Time vs Persistent Shocks\n({SPEC_SHORT_NAME})',
             fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(output_dir / 'figure_15_irf_persistent_comparison.png', dpi=300, bbox_inches='tight')
plt.savefig(output_dir / 'figure_15_irf_persistent_comparison.pdf', bbox_inches='tight')
print(f"  Saved to {output_dir / 'figure_15_irf_persistent_comparison.png'}")
plt.show()


# %%
# =============================================================================
# FIGURE 16 (NEW): ALL SHOCKS COMBINED
# =============================================================================
print("\nCreating Figure 16: All Shocks Combined...")

fig, ax = plt.subplots(figsize=(14, 8))

# Original BB shocks
ax.plot(quarters, irf_energy_f['gcpi_simul'].values, color=colors['Energy'],
        linewidth=2, label='Energy Prices')
ax.plot(quarters, irf_food_f['gcpi_simul'].values, color=colors['Food'],
        linewidth=2, label='Food Prices')
ax.plot(quarters, irf_shortage_f['gcpi_simul'].values, color=colors['Shortage'],
        linewidth=2, label='Shortages (direct)')
ax.plot(quarters, irf_vu_f['gcpi_simul'].values, color=colors['V/U'],
        linewidth=2, label='V/U (persistent)')

# New model shocks
ax.plot(quarters, irf_gscpi_f['gcpi_simul'].values, color=colors['GSCPI'],
        linewidth=2, linestyle='--', label='GSCPI')
ax.plot(quarters, irf_gcu_f['gcpi_simul'].values, color=colors['Capacity Util'],
        linewidth=2, linestyle='--', label='Capacity Util')

ax.set_title(f'Inflation response to all shocks\n(Solid = Original BB, Dashed = New Model)\n({SPEC_SHORT_NAME})',
             fontsize=16, fontweight='normal')
ax.set_xlabel('Quarter', fontsize=16)
ax.set_ylabel('Percent', fontsize=16)

ax.set_xlim(0.5, 16.5)
ax.set_xticks(range(1, 17))

ax.yaxis.grid(True, linestyle='-', linewidth=0.5, color='lightgray')
ax.xaxis.grid(False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.axhline(y=0, color='black', linewidth=0.5)

ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=3,
          frameon=True, edgecolor='black', fancybox=False)

plt.tight_layout()
plt.savefig(output_dir / 'figure_16_irf_all_shocks.png', dpi=300, bbox_inches='tight')
plt.savefig(output_dir / 'figure_16_irf_all_shocks.pdf', bbox_inches='tight')
print(f"  Saved to {output_dir / 'figure_16_irf_all_shocks.png'}")
plt.show()


# %%
# =============================================================================
# FIGURE 17 (NEW): ENDOGENOUS FEEDBACK LOOP - V/U SHOCK
# Shows: vu -> gw -> W -> ed -> shortage -> gcpi
# =============================================================================
print("\nCreating Figure 17: V/U Shock - Endogenous Feedback Loop...")

fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# Panel A: V/U shock series
ax1 = axes[0, 0]
ax1.plot(quarters, irf_vu_f['vu_shock_series'].values, color=colors['V/U'], linewidth=2)
ax1.set_title('(A) V/U Shock', fontsize=12, fontweight='bold')
ax1.set_xlabel('Quarter', fontsize=10)
ax1.set_ylabel('V/U Ratio', fontsize=10)
ax1.yaxis.grid(True, linestyle='-', linewidth=0.5, color='lightgray')
ax1.xaxis.grid(False)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# Panel B: Wage growth
ax2 = axes[0, 1]
ax2.plot(quarters, irf_vu_f['gw_simul'].values, color=colors['V/U'], linewidth=2)
ax2.set_title('(B) Wage Growth Response', fontsize=12, fontweight='bold')
ax2.set_xlabel('Quarter', fontsize=10)
ax2.set_ylabel('Percent', fontsize=10)
ax2.yaxis.grid(True, linestyle='-', linewidth=0.5, color='lightgray')
ax2.xaxis.grid(False)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.axhline(y=0, color='black', linewidth=0.5)

# Panel C: Log wage level
ax3 = axes[0, 2]
ax3.plot(quarters, irf_vu_f['log_w_simul'].values, color=colors['Wage Level'], linewidth=2)
ax3.set_title('(C) Log Wage Level', fontsize=12, fontweight='bold')
ax3.set_xlabel('Quarter', fontsize=10)
ax3.set_ylabel('Log Level', fontsize=10)
ax3.yaxis.grid(True, linestyle='-', linewidth=0.5, color='lightgray')
ax3.xaxis.grid(False)
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)
ax3.axhline(y=0, color='black', linewidth=0.5)

# Panel D: Excess demand (endogenous)
ax4 = axes[1, 0]
ax4.plot(quarters, irf_vu_f['ed_simul'].values, color=colors['Excess Demand'], linewidth=2)
ax4.set_title('(D) Excess Demand (Endogenous)', fontsize=12, fontweight='bold')
ax4.set_xlabel('Quarter', fontsize=10)
ax4.set_ylabel('ED Index', fontsize=10)
ax4.yaxis.grid(True, linestyle='-', linewidth=0.5, color='lightgray')
ax4.xaxis.grid(False)
ax4.spines['top'].set_visible(False)
ax4.spines['right'].set_visible(False)
ax4.axhline(y=0, color='black', linewidth=0.5)

# Panel E: Shortage (endogenous)
ax5 = axes[1, 1]
ax5.plot(quarters, irf_vu_f['shortage_simul'].values, color=colors['Shortage'], linewidth=2)
ax5.set_title('(E) Shortage (Endogenous)', fontsize=12, fontweight='bold')
ax5.set_xlabel('Quarter', fontsize=10)
ax5.set_ylabel('Shortage Index', fontsize=10)
ax5.yaxis.grid(True, linestyle='-', linewidth=0.5, color='lightgray')
ax5.xaxis.grid(False)
ax5.spines['top'].set_visible(False)
ax5.spines['right'].set_visible(False)
ax5.axhline(y=0, color='black', linewidth=0.5)

# Panel F: Inflation
ax6 = axes[1, 2]
ax6.plot(quarters, irf_vu_f['gcpi_simul'].values, color=colors['V/U'], linewidth=2)
ax6.set_title('(F) Inflation Response', fontsize=12, fontweight='bold')
ax6.set_xlabel('Quarter', fontsize=10)
ax6.set_ylabel('Percent', fontsize=10)
ax6.yaxis.grid(True, linestyle='-', linewidth=0.5, color='lightgray')
ax6.xaxis.grid(False)
ax6.spines['top'].set_visible(False)
ax6.spines['right'].set_visible(False)
ax6.axhline(y=0, color='black', linewidth=0.5)

plt.suptitle(f'V/U Shock: Endogenous Feedback Loop\n(vu -> gw -> W -> ed -> shortage -> gcpi)\n({SPEC_SHORT_NAME})',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(output_dir / 'figure_17_irf_vu_feedback_loop.png', dpi=300, bbox_inches='tight')
plt.savefig(output_dir / 'figure_17_irf_vu_feedback_loop.pdf', bbox_inches='tight')
print(f"  Saved to {output_dir / 'figure_17_irf_vu_feedback_loop.png'}")
plt.show()


# %%
# =============================================================================
# FIGURE 18 (NEW): ENDOGENOUS FEEDBACK LOOP - CAPACITY UTILIZATION SHOCK
# Shows: cu -> gw -> W -> ed -> shortage -> gcpi
# =============================================================================
print("\nCreating Figure 18: Capacity Utilization Shock - Endogenous Feedback Loop...")

fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# Panel A: CU shock series
ax1 = axes[0, 0]
ax1.plot(quarters, irf_gcu_f['gcu_shock_series'].values, color=colors['Capacity Util'], linewidth=2)
cu_type = "log(CU)" if USE_LOG_CU_WAGES else "Level CU"
ax1.set_title(f'(A) Capacity Utilization Shock ({cu_type})', fontsize=12, fontweight='bold')
ax1.set_xlabel('Quarter', fontsize=10)
ax1.set_ylabel('CU (detrended)', fontsize=10)
ax1.yaxis.grid(True, linestyle='-', linewidth=0.5, color='lightgray')
ax1.xaxis.grid(False)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# Panel B: Wage growth
ax2 = axes[0, 1]
ax2.plot(quarters, irf_gcu_f['gw_simul'].values, color=colors['Capacity Util'], linewidth=2)
ax2.set_title('(B) Wage Growth Response', fontsize=12, fontweight='bold')
ax2.set_xlabel('Quarter', fontsize=10)
ax2.set_ylabel('Percent', fontsize=10)
ax2.yaxis.grid(True, linestyle='-', linewidth=0.5, color='lightgray')
ax2.xaxis.grid(False)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.axhline(y=0, color='black', linewidth=0.5)

# Panel C: Log wage level
ax3 = axes[0, 2]
ax3.plot(quarters, irf_gcu_f['log_w_simul'].values, color=colors['Wage Level'], linewidth=2)
ax3.set_title('(C) Log Wage Level', fontsize=12, fontweight='bold')
ax3.set_xlabel('Quarter', fontsize=10)
ax3.set_ylabel('Log Level', fontsize=10)
ax3.yaxis.grid(True, linestyle='-', linewidth=0.5, color='lightgray')
ax3.xaxis.grid(False)
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)
ax3.axhline(y=0, color='black', linewidth=0.5)

# Panel D: Excess demand (endogenous)
ax4 = axes[1, 0]
ax4.plot(quarters, irf_gcu_f['ed_simul'].values, color=colors['Excess Demand'], linewidth=2)
ax4.set_title('(D) Excess Demand (Endogenous)', fontsize=12, fontweight='bold')
ax4.set_xlabel('Quarter', fontsize=10)
ax4.set_ylabel('ED Index', fontsize=10)
ax4.yaxis.grid(True, linestyle='-', linewidth=0.5, color='lightgray')
ax4.xaxis.grid(False)
ax4.spines['top'].set_visible(False)
ax4.spines['right'].set_visible(False)
ax4.axhline(y=0, color='black', linewidth=0.5)

# Panel E: Shortage (endogenous)
ax5 = axes[1, 1]
ax5.plot(quarters, irf_gcu_f['shortage_simul'].values, color=colors['Shortage'], linewidth=2)
ax5.set_title('(E) Shortage (Endogenous)', fontsize=12, fontweight='bold')
ax5.set_xlabel('Quarter', fontsize=10)
ax5.set_ylabel('Shortage Index', fontsize=10)
ax5.yaxis.grid(True, linestyle='-', linewidth=0.5, color='lightgray')
ax5.xaxis.grid(False)
ax5.spines['top'].set_visible(False)
ax5.spines['right'].set_visible(False)
ax5.axhline(y=0, color='black', linewidth=0.5)

# Panel F: Inflation
ax6 = axes[1, 2]
ax6.plot(quarters, irf_gcu_f['gcpi_simul'].values, color=colors['Capacity Util'], linewidth=2)
ax6.set_title('(F) Inflation Response', fontsize=12, fontweight='bold')
ax6.set_xlabel('Quarter', fontsize=10)
ax6.set_ylabel('Percent', fontsize=10)
ax6.yaxis.grid(True, linestyle='-', linewidth=0.5, color='lightgray')
ax6.xaxis.grid(False)
ax6.spines['top'].set_visible(False)
ax6.spines['right'].set_visible(False)
ax6.axhline(y=0, color='black', linewidth=0.5)

plt.suptitle(f'Capacity Utilization Shock: Endogenous Feedback Loop\n(cu -> gw -> W -> ed -> shortage -> gcpi)\n({SPEC_SHORT_NAME})',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(output_dir / 'figure_18_irf_cu_feedback_loop.png', dpi=300, bbox_inches='tight')
plt.savefig(output_dir / 'figure_18_irf_cu_feedback_loop.pdf', bbox_inches='tight')
print(f"  Saved to {output_dir / 'figure_18_irf_cu_feedback_loop.png'}")
plt.show()


# %%
# =============================================================================
# FIGURE 19 (NEW): WAGE RESPONSE TO LABOR MARKET SHOCKS
# =============================================================================
print("\nCreating Figure 19: Wage Response to Labor Market Shocks...")

fig, ax = plt.subplots(figsize=(12, 7))

ax.plot(quarters, irf_vu_f['gw_simul'].values, color=colors['V/U'],
        linewidth=2, label='V/U (persistent)')
ax.plot(quarters, irf_gcu_f['gw_simul'].values, color=colors['Capacity Util'],
        linewidth=2, label='Capacity Util (one-time)')
ax.plot(quarters, irf_gcu_persistent_f['gw_simul'].values, color=colors['Capacity Util'],
        linewidth=2, linestyle='--', label='Capacity Util (persistent)')

ax.set_title(f'Wage inflation response to labor market shocks\n({SPEC_SHORT_NAME})',
             fontsize=17.5, fontweight='normal')
ax.set_xlabel('Quarter', fontsize=16)
ax.set_ylabel('Percent', fontsize=16)

ax.set_xlim(0.5, 16.5)
ax.set_xticks(range(1, 17))

ax.yaxis.grid(True, linestyle='-', linewidth=0.5, color='lightgray')
ax.xaxis.grid(False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.axhline(y=0, color='black', linewidth=0.5)

ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.10), ncol=3,
          frameon=True, edgecolor='black', fancybox=False)

plt.tight_layout()
plt.savefig(output_dir / 'figure_19_irf_wage_response.png', dpi=300, bbox_inches='tight')
plt.savefig(output_dir / 'figure_19_irf_wage_response.pdf', bbox_inches='tight')
print(f"  Saved to {output_dir / 'figure_19_irf_wage_response.png'}")
plt.show()


# %%
# =============================================================================
# SUMMARY STATISTICS
# =============================================================================
print("\n" + "=" * 80)
print("IRF SUMMARY")
print("=" * 80)
print(f"Specification: {SPEC_SHORT_NAME}")

print(f"\nOriginal BB Shocks - Peak inflation response:")
print("-" * 60)
print(f"  Energy:     {irf_energy_f['gcpi_simul'].max():.4f} at quarter {irf_energy_f['gcpi_simul'].argmax() + 1}")
print(f"  Food:       {irf_food_f['gcpi_simul'].max():.4f} at quarter {irf_food_f['gcpi_simul'].argmax() + 1}")
print(f"  Shortage:   {irf_shortage_f['gcpi_simul'].max():.4f} at quarter {irf_shortage_f['gcpi_simul'].argmax() + 1}")
print(f"  V/U:        {irf_vu_f['gcpi_simul'].max():.4f} at quarter {irf_vu_f['gcpi_simul'].argmax() + 1}")

print(f"\nNew Model Shocks - Peak inflation response (one-time):")
print("-" * 60)
print(f"  GSCPI:            {irf_gscpi_f['gcpi_simul'].max():.4f} at quarter {irf_gscpi_f['gcpi_simul'].argmax() + 1}")
print(f"  Capacity Util:    {irf_gcu_f['gcpi_simul'].max():.4f} at quarter {irf_gcu_f['gcpi_simul'].argmax() + 1}")
print(f"  Potential GDP:    {irf_ngdppot_f['gcpi_simul'].min():.4f} at quarter {irf_ngdppot_f['gcpi_simul'].argmin() + 1} (deflationary)")

print(f"\nNew Model Shocks - Peak inflation response (persistent):")
print("-" * 60)
print(f"  GSCPI:            {irf_gscpi_persistent_f['gcpi_simul'].max():.4f} at quarter {irf_gscpi_persistent_f['gcpi_simul'].argmax() + 1}")
print(f"  Capacity Util:    {irf_gcu_persistent_f['gcpi_simul'].max():.4f} at quarter {irf_gcu_persistent_f['gcpi_simul'].argmax() + 1}")
print(f"  Potential GDP:    {irf_ngdppot_persistent_f['gcpi_simul'].min():.4f} at quarter {irf_ngdppot_persistent_f['gcpi_simul'].argmin() + 1} (deflationary)")

print(f"\nEndogenous Feedback Loop - V/U Shock:")
print("-" * 60)
print(f"  Peak wage growth:      {irf_vu_f['gw_simul'].max():.4f} at quarter {irf_vu_f['gw_simul'].argmax() + 1}")
print(f"  Peak log wage level:   {irf_vu_f['log_w_simul'].max():.6f} at quarter {irf_vu_f['log_w_simul'].argmax() + 1}")
print(f"  Peak excess demand:    {irf_vu_f['ed_simul'].max():.6f} at quarter {irf_vu_f['ed_simul'].argmax() + 1}")
print(f"  Peak shortage:         {irf_vu_f['shortage_simul'].max():.4f} at quarter {irf_vu_f['shortage_simul'].argmax() + 1}")
print(f"  Peak inflation:        {irf_vu_f['gcpi_simul'].max():.4f} at quarter {irf_vu_f['gcpi_simul'].argmax() + 1}")

print(f"\nEndogenous Feedback Loop - Capacity Utilization Shock:")
print("-" * 60)
print(f"  Peak wage growth:      {irf_gcu_f['gw_simul'].max():.4f} at quarter {irf_gcu_f['gw_simul'].argmax() + 1}")
print(f"  Peak log wage level:   {irf_gcu_f['log_w_simul'].max():.6f} at quarter {irf_gcu_f['log_w_simul'].argmax() + 1}")
print(f"  Peak excess demand:    {irf_gcu_f['ed_simul'].max():.6f} at quarter {irf_gcu_f['ed_simul'].argmax() + 1}")
print(f"  Peak shortage:         {irf_gcu_f['shortage_simul'].max():.4f} at quarter {irf_gcu_f['shortage_simul'].argmax() + 1}")
print(f"  Peak inflation:        {irf_gcu_f['gcpi_simul'].max():.4f} at quarter {irf_gcu_f['gcpi_simul'].argmax() + 1}")

print(f"\nEndogenous Feedback Loop - Potential GDP Shock:")
print("-" * 60)
print(f"  Log NGDPPOT deviation:  {irf_ngdppot_f['log_ngdppot_simul'].max():.6f} at quarter {irf_ngdppot_f['log_ngdppot_simul'].argmax() + 1}")
print(f"  Min excess demand:      {irf_ngdppot_f['ed_simul'].min():.6f} at quarter {irf_ngdppot_f['ed_simul'].argmin() + 1}")
print(f"  Min shortage:           {irf_ngdppot_f['shortage_simul'].min():.4f} at quarter {irf_ngdppot_f['shortage_simul'].argmin() + 1}")
print(f"  Min inflation:          {irf_ngdppot_f['gcpi_simul'].min():.4f} at quarter {irf_ngdppot_f['gcpi_simul'].argmin() + 1}")

print("\n" + "=" * 80)
print("PLOTTING COMPLETE!")
print("=" * 80)
print(f"\nOutput files saved to: {output_dir}")
print("  Original BB:")
print("    - figure_10_irf_supply_shocks.png/pdf")
print("    - figure_11_irf_vu_shock.png/pdf")
print("  New Model:")
print("    - figure_12_irf_gscpi_shock.png/pdf")
print("    - figure_13_irf_capacity_util.png/pdf")
print("    - figure_14_irf_ngdppot_shock.png/pdf")
print("    - figure_15_irf_persistent_comparison.png/pdf")
print("    - figure_16_irf_all_shocks.png/pdf")
print("    - figure_17_irf_vu_feedback_loop.png/pdf")
print("    - figure_18_irf_cu_feedback_loop.png/pdf")
print("    - figure_19_irf_wage_response.png/pdf")
print("\n")

# %%
fig, ax = plt.subplots(figsize=(12, 7))

ax.plot(quarters, irf_gcu_f['cf1_simul'].values, color=colors['V/U'], linewidth=2)

ax.set_title(f'Capacity Utilization Shock: Wage Growth Response\n({SPEC_SHORT_NAME})',
             fontsize=17.5, fontweight='normal')
ax.set_xlabel('Quarter', fontsize=16)
ax.set_ylabel('Percent', fontsize=16)

ax.set_xlim(0.5, 16.5)
ax.set_xticks(range(1, 17))
ax.set_ylim(-2.0, 2.0)

ax.yaxis.grid(True, linestyle='-', linewidth=0.5, color='lightgray')
ax.xaxis.grid(False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
# %%
