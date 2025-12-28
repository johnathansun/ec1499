# %%
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modified Bernanke-Blanchard Model - Decomposition Plots
Liang & Sun (2025)

This file generates decomposition plots for the modified model with
support for multiple model specifications.

Configuration flags at the top control which specification to plot.
The script automatically loads the appropriate decomposition results
based on the configuration.

Generates:
- Figure 12 (New): Sources of Price Inflation (with new channels)
- Figure 13 (New): Sources of Wage Inflation (with capacity utilization)
- Figure 14 (New): Shortage Decomposition (excess demand vs GSCPI)
- Figure 15 (New): Alternative Inflation Decomposition (shortages split into Excess Demand + GSCPI + Capacity Util)
- Figure 16 (New): Side-by-side comparison of Standard vs Alternative decomposition
- Figure 17 (New): Combined 3-panel figure
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# %%
#****************************CONFIGURATION**************************************
# These flags MUST match the specification used in decomp_new_model.py!

USE_PRE_COVID_SAMPLE = False       # Use pre-COVID sample estimates
USE_LOG_CU_WAGES = False          # True = log(CU), False = level CU
USE_CONTEMP_CU = False             # True = CU lags 0-4, False = CU lags 1-4
USE_DETRENDED_EXCESS_DEMAND = True  # Detrend excess demand in shortage eq

#****************************PATH CONFIGURATION*********************************

BASE_DIR = Path("/Users/johnathansun/Documents/ec1499/Replication Package/Code and Data")

# Build directory names based on configuration (must match decomp_new_model.py output)
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
    if USE_DETRENDED_EXCESS_DEMAND:
        parts.append("Detrended ED")
    return ", ".join(parts)

SPEC_DIR_NAME = get_spec_dir_name()
SPEC_SHORT_NAME = get_spec_short_name()

# Input Location - output from decomp_new_model.py
input_dir = BASE_DIR / "(3) Core Results/Decompositions/Output Data Python (New Model)" / SPEC_DIR_NAME

# Output Location for figures
output_dir = BASE_DIR / "(3) Core Results/Decompositions/Figures Python (New Model)" / SPEC_DIR_NAME
output_dir.mkdir(parents=True, exist_ok=True)

#*********************************************************************************

print("="*80)
print("MODIFIED MODEL DECOMPOSITION PLOTS")
print("Liang & Sun (2025)")
print("="*80)

print(f"\nSpecification: {SPEC_SHORT_NAME}")
print(f"  USE_PRE_COVID_SAMPLE:        {USE_PRE_COVID_SAMPLE}")
print(f"  USE_LOG_CU_WAGES:            {USE_LOG_CU_WAGES}")
print(f"  USE_CONTEMP_CU:              {USE_CONTEMP_CU}")
print(f"  USE_DETRENDED_EXCESS_DEMAND: {USE_DETRENDED_EXCESS_DEMAND}")

print(f"\nInput directory:  {input_dir}")
print(f"Output directory: {output_dir}")

print("\nLoading decomposition results...")

# Load decomposition data
baseline = pd.read_excel(input_dir / 'baseline.xlsx')
remove_grpe = pd.read_excel(input_dir / 'remove_grpe.xlsx')
remove_grpf = pd.read_excel(input_dir / 'remove_grpf.xlsx')
remove_vu = pd.read_excel(input_dir / 'remove_vu.xlsx')
remove_short = pd.read_excel(input_dir / 'remove_shortage.xlsx')
remove_magpty = pd.read_excel(input_dir / 'remove_magpty.xlsx')
remove_q2 = pd.read_excel(input_dir / 'remove_2020q2.xlsx')
remove_q3 = pd.read_excel(input_dir / 'remove_2020q3.xlsx')

# NEW model-specific decompositions
remove_excess_demand = pd.read_excel(input_dir / 'remove_excess_demand.xlsx')
remove_gscpi = pd.read_excel(input_dir / 'remove_gscpi.xlsx')
remove_cu = pd.read_excel(input_dir / 'remove_cu.xlsx')
remove_all = pd.read_excel(input_dir / 'remove_all.xlsx')

print(f"Loaded {len(baseline)} observations")

# %%
# Convert period to datetime if needed
for df in [baseline, remove_grpe, remove_grpf, remove_vu, remove_short, remove_magpty,
           remove_q2, remove_q3, remove_excess_demand, remove_gscpi, remove_cu, remove_all]:
    if not pd.api.types.is_datetime64_any_dtype(df['period']):
        df['period'] = pd.to_datetime(df['period'])

# Filter to period >= 2019 Q3 (July 2019)
filter_date = '2019-07-01'
datasets = {
    'baseline': baseline,
    'remove_grpe': remove_grpe,
    'remove_grpf': remove_grpf,
    'remove_vu': remove_vu,
    'remove_short': remove_short,
    'remove_magpty': remove_magpty,
    'remove_q2': remove_q2,
    'remove_q3': remove_q3,
    'remove_excess_demand': remove_excess_demand,
    'remove_gscpi': remove_gscpi,
    'remove_cu': remove_cu,
    'remove_all': remove_all
}

for name, df in datasets.items():
    datasets[name] = df[df['period'] >= filter_date].copy().reset_index(drop=True)

# Unpack filtered datasets
baseline = datasets['baseline']
remove_grpe = datasets['remove_grpe']
remove_grpf = datasets['remove_grpf']
remove_vu = datasets['remove_vu']
remove_short = datasets['remove_short']
remove_magpty = datasets['remove_magpty']
remove_q2 = datasets['remove_q2']
remove_q3 = datasets['remove_q3']
remove_excess_demand = datasets['remove_excess_demand']
remove_gscpi = datasets['remove_gscpi']
remove_cu = datasets['remove_cu']
remove_all = datasets['remove_all']

print(f"Filtered to {len(baseline)} observations from {filter_date}")

# %%
# Helper function to format quarter labels
def period_to_quarter_label(dt):
    """Convert datetime to 'YYYY QN' format"""
    if pd.isna(dt):
        return ''
    return f"{dt.year} Q{(dt.month - 1) // 3 + 1}"

quarter_labels = [period_to_quarter_label(p) for p in baseline['period']]

# %%
# Define colors - colorblind-friendly palette
# Based on Wong (2011) "Points of view: Color blindness" Nature Methods
# and IBM Design accessibility guidelines
# These colors are distinguishable by people with deuteranopia, protanopia, and tritanopia
colors = {
    # Grey for baseline/initial conditions
    'Initial Conditions': '#888888',  # Medium grey

    # Main economic drivers - using high-contrast, colorblind-safe colors
    'V/U': '#D55E00',           # Vermillion (burnt orange) - labor market
    'Energy Prices': '#0072B2',  # Blue - energy
    'Food Prices': '#56B4E9',    # Sky blue - food (lighter than energy)
    'Shortages': '#F0E442',      # Yellow - shortages
    'Productivity': '#E69F00',   # Orange - productivity

    # COVID dummies - using pattern-like contrast
    'Q2 Dummy': '#332288',       # Dark indigo
    'Q3 Dummy': '#AA4499',       # Purple/magenta

    # NEW model variables - distinct from above
    'Capacity Util': '#CC79A7',  # Reddish purple/pink
    'Excess Demand': '#882255',  # Dark magenta/wine
    'GSCPI': '#009E73'           # Bluish green (teal)
}

# Order for stacking (bottom to top) - Initial Conditions at bottom
stack_order_price = ['Initial Conditions', 'Q3 Dummy', 'Q2 Dummy', 'Productivity',
                     'V/U', 'Food Prices', 'Energy Prices', 'Shortages']

# For wage decomposition - includes capacity utilization
stack_order_wage = ['Initial Conditions', 'Q3 Dummy', 'Q2 Dummy', 'Productivity',
                    'Capacity Util', 'V/U', 'Food Prices', 'Energy Prices', 'Shortages']

# For shortage decomposition
stack_order_shortage = ['Initial Conditions', 'Excess Demand', 'GSCPI']

# %%
# Common plot styling
def setup_plot_style():
    """Set up common plot styling"""
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 16,
        'axes.titlesize': 17.5,
        'xtick.labelsize': 12,
        'ytick.labelsize': 14,
        'legend.fontsize': 10,
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'axes.edgecolor': 'black',
        'axes.linewidth': 1,
    })

setup_plot_style()

# %%
# =============================================================================
# FIGURE 12 (NEW): SOURCES OF PRICE INFLATION
# =============================================================================
print("\nCreating Figure 12 (New Model): Sources of Price Inflation...")

# Create decomposition data for prices
decomp_gcpi = pd.DataFrame({
    'period': remove_all['period'],
    'Initial Conditions': remove_all['gcpi_simul'],
    'Energy Prices': remove_grpe['grpe_contr_gcpi'],
    'Food Prices': remove_grpf['grpf_contr_gcpi'],
    'Shortages': remove_short['shortage_contr_gcpi'],
    'V/U': remove_vu['vu_contr_gcpi'],
    'Productivity': remove_magpty['magpty_contr_gcpi'],
    'Q2 Dummy': remove_q2['dummy2020_q2_contr_gcpi'],
    'Q3 Dummy': remove_q3['dummy2020_q3_contr_gcpi']
})

actual_gcpi = baseline['gcpi'].values

fig, ax = plt.subplots(figsize=(14, 8))

x = np.arange(len(decomp_gcpi))
width = 0.6

# Stack bars
bottom_pos = np.zeros(len(decomp_gcpi))
bottom_neg = np.zeros(len(decomp_gcpi))

for component in stack_order_price:
    values = decomp_gcpi[component].values
    pos_vals = np.where(values >= 0, values, 0)
    neg_vals = np.where(values < 0, values, 0)

    ax.bar(x, pos_vals, width, bottom=bottom_pos, label=component,
           color=colors[component], edgecolor='white', linewidth=0.5)
    bottom_pos += pos_vals

    ax.bar(x, neg_vals, width, bottom=bottom_neg,
           color=colors[component], edgecolor='white', linewidth=0.5)
    bottom_neg += neg_vals

# Add actual inflation line
ax.plot(x, actual_gcpi, color='black', linewidth=2, label='Actual Inflation', marker='', zorder=10)

ax.set_title(f'Original sources of price inflation (BB)',
             fontsize=17.5, fontweight='normal')
ax.set_xlabel('Quarter', fontsize=16)
ax.set_ylabel('Percent', fontsize=16)

tick_positions = x[::2]
tick_labels_subset = [quarter_labels[i] for i in tick_positions]
ax.set_xticks(tick_positions)
ax.set_xticklabels(tick_labels_subset, rotation=0, ha='center')

ax.set_ylim(-5, 11)
ax.set_yticks(np.arange(-4, 12, 2))

ax.yaxis.grid(True, linestyle='-', linewidth=0.5, color='lightgray', zorder=0)
ax.xaxis.grid(False)
ax.set_axisbelow(True)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.axhline(y=0, color='black', linewidth=0.5, zorder=1)

legend_order = list(reversed(stack_order_price))
handles = [mpatches.Patch(color=colors[comp], label=comp) for comp in legend_order]
handles.append(plt.Line2D([0], [0], color='black', linewidth=2, label='Actual Inflation'))
ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, -0.10),
          ncol=5, frameon=True, edgecolor='black', fancybox=False)

plt.tight_layout()
plt.savefig(output_dir / 'figure_12_price_decomposition_new.png', dpi=300, bbox_inches='tight')
plt.savefig(output_dir / 'figure_12_price_decomposition_new.pdf', bbox_inches='tight')
print(f"  Saved to {output_dir / 'figure_12_price_decomposition_new.png'}")
plt.show()


# %%
# =============================================================================
# FIGURE 13 (NEW): SOURCES OF WAGE INFLATION (with Capacity Utilization)
# =============================================================================
print("\nCreating Figure 13 (New Model): Sources of Wage Inflation...")

# Create decomposition data for wages - NOW INCLUDES CAPACITY UTILIZATION
decomp_gw = pd.DataFrame({
    'period': remove_all['period'],
    'Initial Conditions': remove_all['gw_simul'],
    'Energy Prices': remove_grpe['grpe_contr_gw'],
    'Food Prices': remove_grpf['grpf_contr_gw'],
    'Shortages': remove_short['shortage_contr_gw'],
    'V/U': remove_vu['vu_contr_gw'],
    'Capacity Util': remove_cu['cu_contr_gw'],  # NEW
    'Productivity': remove_magpty['magpty_contr_gw'],
    'Q2 Dummy': remove_q2['dummy2020_q2_contr_gw'],
    'Q3 Dummy': remove_q3['dummy2020_q3_contr_gw']
})

actual_gw = baseline['gw'].values

fig, ax = plt.subplots(figsize=(14, 8))

bottom_pos = np.zeros(len(decomp_gw))
bottom_neg = np.zeros(len(decomp_gw))

for component in stack_order_wage:
    values = decomp_gw[component].values
    pos_vals = np.where(values >= 0, values, 0)
    neg_vals = np.where(values < 0, values, 0)

    ax.bar(x, pos_vals, width, bottom=bottom_pos, label=component,
           color=colors[component], edgecolor='white', linewidth=0.5)
    bottom_pos += pos_vals

    ax.bar(x, neg_vals, width, bottom=bottom_neg,
           color=colors[component], edgecolor='white', linewidth=0.5)
    bottom_neg += neg_vals

ax.plot(x, actual_gw, color='black', linewidth=2, label='Actual Wage Inflation', marker='', zorder=10)

ax.set_title(f'Sources of wage inflation',
             fontsize=17.5, fontweight='normal')
ax.set_xlabel('Quarter', fontsize=16)
ax.set_ylabel('Percent', fontsize=16)

ax.set_xticks(tick_positions)
ax.set_xticklabels(tick_labels_subset, rotation=0, ha='center')
ax.set_ylim(-6, 8)
ax.set_yticks(np.arange(-5, 8, 2))

ax.yaxis.grid(True, linestyle='-', linewidth=0.5, color='lightgray', zorder=0)
ax.xaxis.grid(False)
ax.set_axisbelow(True)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.axhline(y=0, color='black', linewidth=0.5, zorder=1)

legend_order = list(reversed(stack_order_wage))
handles = [mpatches.Patch(color=colors[comp], label=comp) for comp in legend_order]
handles.append(plt.Line2D([0], [0], color='black', linewidth=2, label='Actual Wage Inflation'))
ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, -0.10),
          ncol=5, frameon=True, edgecolor='black', fancybox=False)

plt.tight_layout()
plt.savefig(output_dir / 'figure_13_wage_decomposition_new.png', dpi=300, bbox_inches='tight')
plt.savefig(output_dir / 'figure_13_wage_decomposition_new.pdf', bbox_inches='tight')
print(f"  Saved to {output_dir / 'figure_13_wage_decomposition_new.png'}")
plt.show()


# %%
# =============================================================================
# FIGURE 14 (NEW): SHORTAGE DECOMPOSITION - Excess Demand vs GSCPI
# =============================================================================
print("\nCreating Figure 14 (New Model): Shortage Decomposition...")

# Create decomposition data for shortages
# Initial conditions = what shortage would be with both channels removed
decomp_shortage = pd.DataFrame({
    'period': baseline['period'],
    'Initial Conditions': remove_all['shortage_simul'],  # Baseline shortage with all shocks removed
    'Excess Demand': remove_excess_demand['excess_demand_contr_shortage'],
    'GSCPI': remove_gscpi['gscpi_contr_shortage']
})

actual_shortage = baseline['shortage'].values
simulated_shortage = baseline['shortage_baseline'].values

fig, ax = plt.subplots(figsize=(14, 8))

bottom_pos = np.zeros(len(decomp_shortage))
bottom_neg = np.zeros(len(decomp_shortage))

for component in stack_order_shortage:
    values = decomp_shortage[component].values
    pos_vals = np.where(values >= 0, values, 0)
    neg_vals = np.where(values < 0, values, 0)

    ax.bar(x, pos_vals, width, bottom=bottom_pos, label=component,
           color=colors[component], edgecolor='white', linewidth=0.5)
    bottom_pos += pos_vals

    ax.bar(x, neg_vals, width, bottom=bottom_neg,
           color=colors[component], edgecolor='white', linewidth=0.5)
    bottom_neg += neg_vals

# Add actual shortage line
ax.plot(x, actual_shortage, color='black', linewidth=2, label='Actual Shortage Index', marker='', zorder=10)
ax.plot(x, simulated_shortage, color='black', linewidth=2, linestyle='--', label='Simulated Shortage', marker='', zorder=10)

ax.set_title(f'Sources of shortage index',
             fontsize=17.5, fontweight='normal')
ax.set_xlabel('Quarter', fontsize=16)
ax.set_ylabel('Shortage Index', fontsize=16)

ax.set_xticks(tick_positions)
ax.set_xticklabels(tick_labels_subset, rotation=0, ha='center')

ax.yaxis.grid(True, linestyle='-', linewidth=0.5, color='lightgray', zorder=0)
ax.xaxis.grid(False)
ax.set_axisbelow(True)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

legend_order = list(reversed(stack_order_shortage))
handles = [mpatches.Patch(color=colors[comp], label=comp) for comp in legend_order]
handles.append(plt.Line2D([0], [0], color='black', linewidth=2, label='Actual Shortage'))
handles.append(plt.Line2D([0], [0], color='black', linewidth=2, linestyle='--', label='Simulated Shortage'))
ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, -0.10),
          ncol=5, frameon=True, edgecolor='black', fancybox=False)

plt.tight_layout()
plt.savefig(output_dir / 'figure_14_shortage_decomposition.png', dpi=300, bbox_inches='tight')
plt.savefig(output_dir / 'figure_14_shortage_decomposition.pdf', bbox_inches='tight')
print(f"  Saved to {output_dir / 'figure_14_shortage_decomposition.png'}")
plt.show()


# %%
# =============================================================================
# FIGURE 15 (NEW): ALTERNATIVE INFLATION DECOMPOSITION
# Replaces "Shortages" with Excess Demand + GSCPI + Capacity Utilization
# =============================================================================
print("\nCreating Figure 15 (New Model): Alternative Inflation Decomposition...")
print("  (Decomposes shortage effect into Excess Demand vs GSCPI)")

# Create alternative decomposition - replace Shortages with its components
decomp_gcpi_alt = pd.DataFrame({
    'period': remove_all['period'],
    'Initial Conditions': remove_all['gcpi_simul'],
    'Energy Prices': remove_grpe['grpe_contr_gcpi'],
    'Food Prices': remove_grpf['grpf_contr_gcpi'],
    'Excess Demand': remove_excess_demand['excess_demand_contr_gcpi'],  # NEW: replaces shortages
    'GSCPI': remove_gscpi['gscpi_contr_gcpi'],  # NEW: replaces shortages
    'Capacity Util': remove_cu['cu_contr_gcpi'],  # NEW: capacity utilization effect
    'V/U': remove_vu['vu_contr_gcpi'],
    'Productivity': remove_magpty['magpty_contr_gcpi'],
    'Q2 Dummy': remove_q2['dummy2020_q2_contr_gcpi'],
    'Q3 Dummy': remove_q3['dummy2020_q3_contr_gcpi']
})

# Stack order for alternative decomposition
stack_order_alt = ['Initial Conditions', 'Q3 Dummy', 'Q2 Dummy', 'Productivity',
                   'Capacity Util', 'V/U', 'Food Prices', 'Energy Prices', 'GSCPI', 'Excess Demand']

fig, ax = plt.subplots(figsize=(14, 8))

bottom_pos = np.zeros(len(decomp_gcpi_alt))
bottom_neg = np.zeros(len(decomp_gcpi_alt))

for component in stack_order_alt:
    values = decomp_gcpi_alt[component].values
    pos_vals = np.where(values >= 0, values, 0)
    neg_vals = np.where(values < 0, values, 0)

    ax.bar(x, pos_vals, width, bottom=bottom_pos, label=component,
           color=colors[component], edgecolor='white', linewidth=0.5)
    bottom_pos += pos_vals

    ax.bar(x, neg_vals, width, bottom=bottom_neg,
           color=colors[component], edgecolor='white', linewidth=0.5)
    bottom_neg += neg_vals

# Add actual inflation line
ax.plot(x, actual_gcpi, color='black', linewidth=2, label='Actual Inflation', marker='', zorder=10)

ax.set_title(f'Sources of price inflation',
             fontsize=16, fontweight='normal')
ax.set_xlabel('Quarter', fontsize=16)
ax.set_ylabel('Percent', fontsize=16)

ax.set_xticks(tick_positions)
ax.set_xticklabels(tick_labels_subset, rotation=0, ha='center')

ax.set_ylim(-5, 11)
ax.set_yticks(np.arange(-4, 12, 2))

ax.yaxis.grid(True, linestyle='-', linewidth=0.5, color='lightgray', zorder=0)
ax.xaxis.grid(False)
ax.set_axisbelow(True)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.axhline(y=0, color='black', linewidth=0.5, zorder=1)

legend_order_alt = list(reversed(stack_order_alt))
handles = [mpatches.Patch(color=colors[comp], label=comp) for comp in legend_order_alt]
handles.append(plt.Line2D([0], [0], color='black', linewidth=2, label='Actual Inflation'))
ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, -0.10),
          ncol=5, frameon=True, edgecolor='black', fancybox=False)

plt.tight_layout()
plt.savefig(output_dir / 'figure_15_alt_inflation_decomposition.png', dpi=300, bbox_inches='tight')
plt.savefig(output_dir / 'figure_15_alt_inflation_decomposition.pdf', bbox_inches='tight')
print(f"  Saved to {output_dir / 'figure_15_alt_inflation_decomposition.png'}")
plt.show()


# %%
# =============================================================================
# FIGURE 16 (NEW): SIDE-BY-SIDE COMPARISON - Original vs Alternative Decomposition
# =============================================================================
print("\nCreating Figure 16: Side-by-Side Comparison...")

fig, axes = plt.subplots(1, 2, figsize=(18, 7))

# Left panel: Original decomposition (with Shortages)
ax1 = axes[0]
bottom_pos = np.zeros(len(decomp_gcpi))
bottom_neg = np.zeros(len(decomp_gcpi))

for component in stack_order_price:
    values = decomp_gcpi[component].values
    pos_vals = np.where(values >= 0, values, 0)
    neg_vals = np.where(values < 0, values, 0)
    ax1.bar(x, pos_vals, width, bottom=bottom_pos, color=colors[component], edgecolor='white', linewidth=0.3)
    bottom_pos += pos_vals
    ax1.bar(x, neg_vals, width, bottom=bottom_neg, color=colors[component], edgecolor='white', linewidth=0.3)
    bottom_neg += neg_vals

ax1.plot(x, actual_gcpi, color='black', linewidth=2, zorder=10)
ax1.set_title('(A) Standard Decomposition\n(Shortages as single component)', fontsize=14, fontweight='bold')
ax1.set_xlabel('Quarter', fontsize=12)
ax1.set_ylabel('Percent', fontsize=12)
ax1.set_xticks(tick_positions)
ax1.set_xticklabels(tick_labels_subset, rotation=0, ha='center', fontsize=10)
ax1.set_ylim(-5, 11)
ax1.yaxis.grid(True, linestyle='-', linewidth=0.5, color='lightgray', zorder=0)
ax1.set_axisbelow(True)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.axhline(y=0, color='black', linewidth=0.5, zorder=1)

# Right panel: Alternative decomposition (Excess Demand + GSCPI + Capacity Util)
ax2 = axes[1]
bottom_pos = np.zeros(len(decomp_gcpi_alt))
bottom_neg = np.zeros(len(decomp_gcpi_alt))

for component in stack_order_alt:
    values = decomp_gcpi_alt[component].values
    pos_vals = np.where(values >= 0, values, 0)
    neg_vals = np.where(values < 0, values, 0)
    ax2.bar(x, pos_vals, width, bottom=bottom_pos, color=colors[component], edgecolor='white', linewidth=0.3)
    bottom_pos += pos_vals
    ax2.bar(x, neg_vals, width, bottom=bottom_neg, color=colors[component], edgecolor='white', linewidth=0.3)
    bottom_neg += neg_vals

ax2.plot(x, actual_gcpi, color='black', linewidth=2, zorder=10)
ax2.set_title('(B) Alternative Decomposition\n(Shortages split into Excess Demand + GSCPI)', fontsize=14, fontweight='bold')
ax2.set_xlabel('Quarter', fontsize=12)
ax2.set_ylabel('Percent', fontsize=12)
ax2.set_xticks(tick_positions)
ax2.set_xticklabels(tick_labels_subset, rotation=0, ha='center', fontsize=10)
ax2.set_ylim(-4, 12)
ax2.yaxis.grid(True, linestyle='-', linewidth=0.5, color='lightgray', zorder=0)
ax2.set_axisbelow(True)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.axhline(y=0, color='black', linewidth=0.5, zorder=1)

# Combined legend
all_components_comparison = list(set(stack_order_price + stack_order_alt))
legend_order_comparison = ['Shortages', 'Excess Demand', 'GSCPI', 'Energy Prices', 'Food Prices',
                           'V/U', 'Capacity Util', 'Productivity', 'Q2 Dummy', 'Q3 Dummy', 'Initial Conditions']
legend_order_comparison = [c for c in legend_order_comparison if c in all_components_comparison]

handles = [mpatches.Patch(color=colors[comp], label=comp) for comp in legend_order_comparison]
handles.append(plt.Line2D([0], [0], color='black', linewidth=2, label='Actual'))

fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 0.02),
           ncol=6, frameon=True, edgecolor='black', fancybox=False, fontsize=10)

plt.suptitle(f'Inflation Decomposition: Standard vs Alternative\n({SPEC_SHORT_NAME})',
             fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(output_dir / 'figure_16_comparison_decomposition.png', dpi=300, bbox_inches='tight')
plt.savefig(output_dir / 'figure_16_comparison_decomposition.pdf', bbox_inches='tight')
print(f"  Saved to {output_dir / 'figure_16_comparison_decomposition.png'}")
plt.show()


# %%
# =============================================================================
# FIGURE 17 (NEW): COMBINED 3-PANEL FIGURE
# =============================================================================
print("\nCreating Figure 17: Combined 3-Panel Decomposition...")

fig, axes = plt.subplots(1, 3, figsize=(20, 7))

# Panel 1: Price Decomposition
ax1 = axes[0]
bottom_pos = np.zeros(len(decomp_gcpi))
bottom_neg = np.zeros(len(decomp_gcpi))

for component in stack_order_price:
    values = decomp_gcpi[component].values
    pos_vals = np.where(values >= 0, values, 0)
    neg_vals = np.where(values < 0, values, 0)
    ax1.bar(x, pos_vals, width, bottom=bottom_pos, color=colors[component], edgecolor='white', linewidth=0.3)
    bottom_pos += pos_vals
    ax1.bar(x, neg_vals, width, bottom=bottom_neg, color=colors[component], edgecolor='white', linewidth=0.3)
    bottom_neg += neg_vals

ax1.plot(x, actual_gcpi, color='black', linewidth=2, zorder=10)
ax1.set_title('(A) Price Inflation', fontsize=14, fontweight='bold')
ax1.set_xlabel('Quarter', fontsize=12)
ax1.set_ylabel('Percent', fontsize=12)
ax1.set_xticks(tick_positions)
ax1.set_xticklabels(tick_labels_subset, rotation=0, ha='center', fontsize=9)
ax1.set_ylim(-4, 12)
ax1.yaxis.grid(True, linestyle='-', linewidth=0.5, color='lightgray', zorder=0)
ax1.set_axisbelow(True)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.axhline(y=0, color='black', linewidth=0.5, zorder=1)

# Panel 2: Wage Decomposition
ax2 = axes[1]
bottom_pos = np.zeros(len(decomp_gw))
bottom_neg = np.zeros(len(decomp_gw))

for component in stack_order_wage:
    values = decomp_gw[component].values
    pos_vals = np.where(values >= 0, values, 0)
    neg_vals = np.where(values < 0, values, 0)
    ax2.bar(x, pos_vals, width, bottom=bottom_pos, color=colors[component], edgecolor='white', linewidth=0.3)
    bottom_pos += pos_vals
    ax2.bar(x, neg_vals, width, bottom=bottom_neg, color=colors[component], edgecolor='white', linewidth=0.3)
    bottom_neg += neg_vals

ax2.plot(x, actual_gw, color='black', linewidth=2, zorder=10)
ax2.set_title('(B) Wage Inflation', fontsize=14, fontweight='bold')
ax2.set_xlabel('Quarter', fontsize=12)
ax2.set_ylabel('Percent', fontsize=12)
ax2.set_xticks(tick_positions)
ax2.set_xticklabels(tick_labels_subset, rotation=0, ha='center', fontsize=9)
ax2.set_ylim(-4, 10)
ax2.yaxis.grid(True, linestyle='-', linewidth=0.5, color='lightgray', zorder=0)
ax2.set_axisbelow(True)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.axhline(y=0, color='black', linewidth=0.5, zorder=1)

# Panel 3: Shortage Decomposition
ax3 = axes[2]
bottom_pos = np.zeros(len(decomp_shortage))
bottom_neg = np.zeros(len(decomp_shortage))

for component in stack_order_shortage:
    values = decomp_shortage[component].values
    pos_vals = np.where(values >= 0, values, 0)
    neg_vals = np.where(values < 0, values, 0)
    ax3.bar(x, pos_vals, width, bottom=bottom_pos, color=colors[component], edgecolor='white', linewidth=0.3)
    bottom_pos += pos_vals
    ax3.bar(x, neg_vals, width, bottom=bottom_neg, color=colors[component], edgecolor='white', linewidth=0.3)
    bottom_neg += neg_vals

ax3.plot(x, actual_shortage, color='black', linewidth=2, zorder=10)
ax3.set_title('(C) Shortage Index', fontsize=14, fontweight='bold')
ax3.set_xlabel('Quarter', fontsize=12)
ax3.set_ylabel('Index', fontsize=12)
ax3.set_xticks(tick_positions)
ax3.set_xticklabels(tick_labels_subset, rotation=0, ha='center', fontsize=9)
ax3.yaxis.grid(True, linestyle='-', linewidth=0.5, color='lightgray', zorder=0)
ax3.set_axisbelow(True)
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)

# Combined legend
all_components = list(set(stack_order_price + stack_order_wage + stack_order_shortage))
# Order by importance
legend_components = ['Shortages', 'Excess Demand', 'GSCPI', 'Energy Prices', 'Food Prices',
                     'V/U', 'Capacity Util', 'Productivity', 'Q2 Dummy', 'Q3 Dummy', 'Initial Conditions']
legend_components = [c for c in legend_components if c in all_components]

handles = [mpatches.Patch(color=colors[comp], label=comp) for comp in legend_components]
handles.append(plt.Line2D([0], [0], color='black', linewidth=2, label='Actual'))

fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 0.02),
           ncol=6, frameon=True, edgecolor='black', fancybox=False, fontsize=10)

plt.suptitle(f'Decomposition of Pandemic-Era Inflation\n({SPEC_SHORT_NAME})',
             fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(output_dir / 'figure_17_combined_decomposition.png', dpi=300, bbox_inches='tight')
plt.savefig(output_dir / 'figure_17_combined_decomposition.pdf', bbox_inches='tight')
print(f"  Saved to {output_dir / 'figure_17_combined_decomposition.png'}")
plt.show()


# %%
# =============================================================================
# SUMMARY STATISTICS
# =============================================================================
print("\n" + "="*80)
print("DECOMPOSITION SUMMARY")
print("="*80)

print(f"\nPeak contributions to PRICE inflation (gcpi) - Standard Decomposition:")
print("-"*60)
print(f"  Energy Prices:      {decomp_gcpi['Energy Prices'].max():.2f} (Q: {quarter_labels[decomp_gcpi['Energy Prices'].argmax()]})")
print(f"  Food Prices:        {decomp_gcpi['Food Prices'].max():.2f} (Q: {quarter_labels[decomp_gcpi['Food Prices'].argmax()]})")
print(f"  Shortages:          {decomp_gcpi['Shortages'].max():.2f} (Q: {quarter_labels[decomp_gcpi['Shortages'].argmax()]})")
print(f"  V/U:                {decomp_gcpi['V/U'].max():.2f} (Q: {quarter_labels[decomp_gcpi['V/U'].argmax()]})")
print(f"  Productivity:       {decomp_gcpi['Productivity'].max():.2f} (Q: {quarter_labels[decomp_gcpi['Productivity'].argmax()]})")

print(f"\nPeak contributions to PRICE inflation (gcpi) - Alternative Decomposition:")
print("-"*60)
print(f"  Energy Prices:      {decomp_gcpi_alt['Energy Prices'].max():.2f} (Q: {quarter_labels[decomp_gcpi_alt['Energy Prices'].argmax()]})")
print(f"  Food Prices:        {decomp_gcpi_alt['Food Prices'].max():.2f} (Q: {quarter_labels[decomp_gcpi_alt['Food Prices'].argmax()]})")
print(f"  Excess Demand:      {decomp_gcpi_alt['Excess Demand'].max():.2f} (Q: {quarter_labels[decomp_gcpi_alt['Excess Demand'].argmax()]}) [NEW]")
print(f"  GSCPI:              {decomp_gcpi_alt['GSCPI'].max():.2f} (Q: {quarter_labels[decomp_gcpi_alt['GSCPI'].argmax()]}) [NEW]")
print(f"  Capacity Util:      {decomp_gcpi_alt['Capacity Util'].max():.2f} (Q: {quarter_labels[decomp_gcpi_alt['Capacity Util'].argmax()]}) [NEW]")
print(f"  V/U:                {decomp_gcpi_alt['V/U'].max():.2f} (Q: {quarter_labels[decomp_gcpi_alt['V/U'].argmax()]})")

print(f"\nPeak contributions to WAGE inflation (gw):")
print("-"*60)
print(f"  V/U:                {decomp_gw['V/U'].max():.2f} (Q: {quarter_labels[decomp_gw['V/U'].argmax()]})")
print(f"  Capacity Util:      {decomp_gw['Capacity Util'].max():.2f} (Q: {quarter_labels[decomp_gw['Capacity Util'].argmax()]}) [NEW]")
print(f"  Energy Prices:      {decomp_gw['Energy Prices'].max():.2f} (Q: {quarter_labels[decomp_gw['Energy Prices'].argmax()]})")
print(f"  Shortages:          {decomp_gw['Shortages'].max():.2f} (Q: {quarter_labels[decomp_gw['Shortages'].argmax()]})")

print(f"\nPeak contributions to SHORTAGE index:")
print("-"*60)
print(f"  Excess Demand:      {decomp_shortage['Excess Demand'].max():.2f} (Q: {quarter_labels[decomp_shortage['Excess Demand'].argmax()]}) [NEW]")
print(f"  GSCPI:              {decomp_shortage['GSCPI'].max():.2f} (Q: {quarter_labels[decomp_shortage['GSCPI'].argmax()]}) [NEW]")

# Calculate relative contributions
print("\n" + "="*80)
print("KEY INSIGHT: SHORTAGE ATTRIBUTION")
print("="*80)

# Sum of contributions over COVID period (2020+)
covid_mask = baseline['period'] >= '2020-01-01'
total_excess_demand = decomp_shortage.loc[covid_mask, 'Excess Demand'].sum()
total_gscpi = decomp_shortage.loc[covid_mask, 'GSCPI'].sum()
total_shortage_change = total_excess_demand + total_gscpi

if abs(total_shortage_change) > 0.01:
    pct_excess_demand = 100 * total_excess_demand / total_shortage_change
    pct_gscpi = 100 * total_gscpi / total_shortage_change
    print(f"\nCumulative shortage attribution (2020+):")
    print(f"  Excess Demand (wages/capacity): {pct_excess_demand:.1f}%")
    print(f"  Supply Chain Pressure (GSCPI):  {pct_gscpi:.1f}%")
    print(f"\nInterpretation:")
    if pct_excess_demand > pct_gscpi:
        print(f"  -> Demand-side factors were the PRIMARY driver of shortages")
    else:
        print(f"  -> Supply-chain factors were the PRIMARY driver of shortages")


# %%
# =============================================================================
# FIGURE 18 (NEW): EXCESS DEMAND COMPONENT ATTRIBUTION TO SHORTAGES
# Shows how wages, capacity utilization, and potential GDP contribute to
# the excess demand component of shortages
# =============================================================================
print("\nCreating Figure 18: Excess Demand Components - Contribution to Shortages...")

# Try to load the component decomposition data
try:
    component_decomp = pd.read_excel(input_dir / 'excess_demand_components.xlsx')
    if not pd.api.types.is_datetime64_any_dtype(component_decomp['period']):
        component_decomp['period'] = pd.to_datetime(component_decomp['period'])
    component_decomp = component_decomp[component_decomp['period'] >= filter_date].copy().reset_index(drop=True)
    has_component_data = True
    print(f"  Loaded component decomposition data: {len(component_decomp)} observations")
except FileNotFoundError:
    print("  WARNING: excess_demand_components.xlsx not found. Run decomp_new_model.py first.")
    has_component_data = False

if has_component_data:
    # Define colors for excess demand components
    ed_component_colors = {
        'Wages': '#D55E00',           # Vermillion
        'Capacity Util': '#0072B2',   # Blue
        'Potential GDP': '#009E73',   # Teal
        'Initial Conditions': '#888888'
    }

    # Stack order for ED components
    stack_order_ed = ['Initial Conditions', 'Potential GDP', 'Capacity Util', 'Wages']

    # Create decomposition of excess demand contribution to shortage
    # The excess demand contribution to shortage is split by its components
    decomp_ed_to_shortage = pd.DataFrame({
        'period': component_decomp['period'],
        'Initial Conditions': np.zeros(len(component_decomp)),  # No initial conditions for this
        'Wages': component_decomp['wage_contr_shortage'],
        'Capacity Util': component_decomp['cu_contr_shortage'],
        'Potential GDP': component_decomp['ngdppot_contr_shortage']
    })

    fig, ax = plt.subplots(figsize=(14, 8))

    x = np.arange(len(decomp_ed_to_shortage))
    width = 0.6

    bottom_pos = np.zeros(len(decomp_ed_to_shortage))
    bottom_neg = np.zeros(len(decomp_ed_to_shortage))

    for component in stack_order_ed:
        values = decomp_ed_to_shortage[component].values
        values = np.nan_to_num(values, nan=0.0)
        pos_vals = np.where(values >= 0, values, 0)
        neg_vals = np.where(values < 0, values, 0)

        ax.bar(x, pos_vals, width, bottom=bottom_pos, label=component,
               color=ed_component_colors[component], edgecolor='white', linewidth=0.5)
        bottom_pos += pos_vals

        ax.bar(x, neg_vals, width, bottom=bottom_neg,
               color=ed_component_colors[component], edgecolor='white', linewidth=0.5)
        bottom_neg += neg_vals

    # Add total excess demand contribution line
    total_ed_contr = component_decomp['ed_contr_shortage'].values
    ax.plot(x, total_ed_contr, color='black', linewidth=2, label='Total Excess Demand Contribution',
            marker='', zorder=10)

    ax.set_title('Excess demand contribution to shortages\n(decomposed by wages, capacity utilization, potential GDP)',
                 fontsize=15, fontweight='normal')
    ax.set_xlabel('Quarter', fontsize=16)
    ax.set_ylabel('Shortage Index Points', fontsize=16)

    tick_positions_ed = x[::2]
    tick_labels_ed = [period_to_quarter_label(p) for p in component_decomp['period']]
    tick_labels_subset_ed = [tick_labels_ed[i] for i in tick_positions_ed]
    ax.set_xticks(tick_positions_ed)
    ax.set_xticklabels(tick_labels_subset_ed, rotation=0, ha='center')

    ax.yaxis.grid(True, linestyle='-', linewidth=0.5, color='lightgray', zorder=0)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.axhline(y=0, color='black', linewidth=0.5, zorder=1)

    legend_order_ed = list(reversed(stack_order_ed[1:]))  # Skip initial conditions
    handles = [mpatches.Patch(color=ed_component_colors[comp], label=comp) for comp in legend_order_ed]
    handles.append(plt.Line2D([0], [0], color='black', linewidth=2, label='Total ED Contribution'))
    ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, -0.10),
              ncol=4, frameon=True, edgecolor='black', fancybox=False)

    plt.tight_layout()
    plt.savefig(output_dir / 'figure_18_ed_components_shortage.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'figure_18_ed_components_shortage.pdf', bbox_inches='tight')
    print(f"  Saved to {output_dir / 'figure_18_ed_components_shortage.png'}")
    plt.show()


# %%
# =============================================================================
# FIGURE 19 (NEW): EXCESS DEMAND COMPONENT ATTRIBUTION TO INFLATION
# Shows how wages, capacity utilization, and potential GDP contribute to
# inflation through the excess demand → shortage → inflation channel
# =============================================================================
if has_component_data:
    print("\nCreating Figure 19: Excess Demand Components - Contribution to Inflation...")

    # Create decomposition of excess demand contribution to inflation
    decomp_ed_to_inflation = pd.DataFrame({
        'period': component_decomp['period'],
        'Initial Conditions': np.zeros(len(component_decomp)),
        'Wages': component_decomp['wage_contr_gcpi_via_ed'],
        'Capacity Util': component_decomp['cu_contr_gcpi_via_ed'],
        'Potential GDP': component_decomp['ngdppot_contr_gcpi_via_ed']
    })

    fig, ax = plt.subplots(figsize=(14, 8))

    bottom_pos = np.zeros(len(decomp_ed_to_inflation))
    bottom_neg = np.zeros(len(decomp_ed_to_inflation))

    for component in stack_order_ed:
        values = decomp_ed_to_inflation[component].values
        values = np.nan_to_num(values, nan=0.0)
        pos_vals = np.where(values >= 0, values, 0)
        neg_vals = np.where(values < 0, values, 0)

        ax.bar(x, pos_vals, width, bottom=bottom_pos, label=component,
               color=ed_component_colors[component], edgecolor='white', linewidth=0.5)
        bottom_pos += pos_vals

        ax.bar(x, neg_vals, width, bottom=bottom_neg,
               color=ed_component_colors[component], edgecolor='white', linewidth=0.5)
        bottom_neg += neg_vals

    # Add total excess demand contribution to inflation line
    total_ed_contr_gcpi = component_decomp['ed_contr_gcpi'].values
    ax.plot(x, total_ed_contr_gcpi, color='black', linewidth=2,
            label='Total Excess Demand Contribution', marker='', zorder=10)

    ax.set_title('Excess demand contribution to inflation via shortages',
                 fontsize=15, fontweight='normal')
    ax.set_xlabel('Quarter', fontsize=16)
    ax.set_ylabel('Percent', fontsize=16)

    ax.set_xticks(tick_positions_ed)
    ax.set_xticklabels(tick_labels_subset_ed, rotation=0, ha='center')

    ax.yaxis.grid(True, linestyle='-', linewidth=0.5, color='lightgray', zorder=0)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.axhline(y=0, color='black', linewidth=0.5, zorder=1)

    handles = [mpatches.Patch(color=ed_component_colors[comp], label=comp) for comp in legend_order_ed]
    handles.append(plt.Line2D([0], [0], color='black', linewidth=2, label='Total ED Contribution'))
    ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, -0.10),
              ncol=4, frameon=True, edgecolor='black', fancybox=False)

    plt.tight_layout()
    plt.savefig(output_dir / 'figure_19_ed_components_inflation.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'figure_19_ed_components_inflation.pdf', bbox_inches='tight')
    print(f"  Saved to {output_dir / 'figure_19_ed_components_inflation.png'}")
    plt.show()


# %%
# =============================================================================
# FIGURE 20 (NEW): CAPACITY UTILIZATION TOTAL EFFECT ON INFLATION
# Shows both direct effect (via wages) and indirect effect (via excess demand → shortage)
# =============================================================================
if has_component_data:
    print("\nCreating Figure 20: Capacity Utilization Total Effect on Inflation...")

    # Capacity utilization affects inflation through two channels:
    # 1. Direct: cu → wages → inflation  (remove_cu['cu_contr_gcpi'])
    # 2. Indirect: cu → excess_demand → shortage → inflation (cu_contr_gcpi_via_ed)

    decomp_cu_inflation = pd.DataFrame({
        'period': component_decomp['period'],
        'Direct (via wages)': component_decomp['cu_contr_gcpi_direct'],
        'Indirect (via shortages)': component_decomp['cu_contr_gcpi_via_ed']
    })

    cu_colors = {
        'Direct (via wages)': '#0072B2',        # Blue
        'Indirect (via shortages)': '#56B4E9'   # Sky blue
    }

    stack_order_cu = ['Direct (via wages)', 'Indirect (via shortages)']

    fig, ax = plt.subplots(figsize=(14, 8))

    bottom_pos = np.zeros(len(decomp_cu_inflation))
    bottom_neg = np.zeros(len(decomp_cu_inflation))

    for component in stack_order_cu:
        values = decomp_cu_inflation[component].values
        values = np.nan_to_num(values, nan=0.0)
        pos_vals = np.where(values >= 0, values, 0)
        neg_vals = np.where(values < 0, values, 0)

        ax.bar(x, pos_vals, width, bottom=bottom_pos, label=component,
               color=cu_colors[component], edgecolor='white', linewidth=0.5)
        bottom_pos += pos_vals

        ax.bar(x, neg_vals, width, bottom=bottom_neg,
               color=cu_colors[component], edgecolor='white', linewidth=0.5)
        bottom_neg += neg_vals

    # Add total capacity utilization contribution
    total_cu_contr = component_decomp['cu_total_contr_gcpi'].values
    ax.plot(x, total_cu_contr, color='black', linewidth=2,
            label='Total Capacity Util Effect', marker='', zorder=10)

    ax.set_title('Total effect of capacity utilization on inflation',
                 fontsize=15, fontweight='normal')
    ax.set_xlabel('Quarter', fontsize=16)
    ax.set_ylabel('Percent', fontsize=16)

    ax.set_xticks(tick_positions_ed)
    ax.set_xticklabels(tick_labels_subset_ed, rotation=0, ha='center')

    ax.yaxis.grid(True, linestyle='-', linewidth=0.5, color='lightgray', zorder=0)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.axhline(y=0, color='black', linewidth=0.5, zorder=1)

    legend_order_cu = list(reversed(stack_order_cu))
    handles = [mpatches.Patch(color=cu_colors[comp], label=comp) for comp in legend_order_cu]
    handles.append(plt.Line2D([0], [0], color='black', linewidth=2, label='Total CU Effect'))
    ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, -0.10),
              ncol=3, frameon=True, edgecolor='black', fancybox=False)

    plt.tight_layout()
    plt.savefig(output_dir / 'figure_20_cu_total_inflation.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'figure_20_cu_total_inflation.pdf', bbox_inches='tight')
    print(f"  Saved to {output_dir / 'figure_20_cu_total_inflation.png'}")
    plt.show()


# %%
# =============================================================================
# FIGURE 21 (NEW): COMBINED 2-PANEL - ED Components to Shortage and Inflation
# =============================================================================
if has_component_data:
    print("\nCreating Figure 21: Combined ED Components Panel...")

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # Left panel: ED components to shortages
    ax1 = axes[0]
    bottom_pos = np.zeros(len(decomp_ed_to_shortage))
    bottom_neg = np.zeros(len(decomp_ed_to_shortage))

    for component in stack_order_ed[1:]:  # Skip initial conditions
        values = decomp_ed_to_shortage[component].values
        values = np.nan_to_num(values, nan=0.0)
        pos_vals = np.where(values >= 0, values, 0)
        neg_vals = np.where(values < 0, values, 0)
        ax1.bar(x, pos_vals, width, bottom=bottom_pos, color=ed_component_colors[component], edgecolor='white', linewidth=0.3)
        bottom_pos += pos_vals
        ax1.bar(x, neg_vals, width, bottom=bottom_neg, color=ed_component_colors[component], edgecolor='white', linewidth=0.3)
        bottom_neg += neg_vals

    ax1.plot(x, component_decomp['ed_contr_shortage'].values, color='black', linewidth=2, zorder=10)
    ax1.set_title('(A) Excess Demand → Shortages', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Quarter', fontsize=12)
    ax1.set_ylabel('Shortage Index Points', fontsize=12)
    ax1.set_xticks(tick_positions_ed)
    ax1.set_xticklabels(tick_labels_subset_ed, rotation=0, ha='center', fontsize=10)
    ax1.yaxis.grid(True, linestyle='-', linewidth=0.5, color='lightgray', zorder=0)
    ax1.set_axisbelow(True)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.axhline(y=0, color='black', linewidth=0.5, zorder=1)

    # Right panel: ED components to inflation
    ax2 = axes[1]
    bottom_pos = np.zeros(len(decomp_ed_to_inflation))
    bottom_neg = np.zeros(len(decomp_ed_to_inflation))

    for component in stack_order_ed[1:]:  # Skip initial conditions
        values = decomp_ed_to_inflation[component].values
        values = np.nan_to_num(values, nan=0.0)
        pos_vals = np.where(values >= 0, values, 0)
        neg_vals = np.where(values < 0, values, 0)
        ax2.bar(x, pos_vals, width, bottom=bottom_pos, color=ed_component_colors[component], edgecolor='white', linewidth=0.3)
        bottom_pos += pos_vals
        ax2.bar(x, neg_vals, width, bottom=bottom_neg, color=ed_component_colors[component], edgecolor='white', linewidth=0.3)
        bottom_neg += neg_vals

    ax2.plot(x, component_decomp['ed_contr_gcpi'].values, color='black', linewidth=2, zorder=10)
    ax2.set_title('(B) Excess Demand → Inflation (via Shortages)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Quarter', fontsize=12)
    ax2.set_ylabel('Percent', fontsize=12)
    ax2.set_xticks(tick_positions_ed)
    ax2.set_xticklabels(tick_labels_subset_ed, rotation=0, ha='center', fontsize=10)
    ax2.yaxis.grid(True, linestyle='-', linewidth=0.5, color='lightgray', zorder=0)
    ax2.set_axisbelow(True)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.axhline(y=0, color='black', linewidth=0.5, zorder=1)

    # Combined legend
    legend_comps = ['Wages', 'Capacity Util', 'Potential GDP']
    handles = [mpatches.Patch(color=ed_component_colors[comp], label=comp) for comp in legend_comps]
    handles.append(plt.Line2D([0], [0], color='black', linewidth=2, label='Total'))

    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 0.02),
               ncol=4, frameon=True, edgecolor='black', fancybox=False, fontsize=11)

    plt.suptitle(f'Decomposition of Excess Demand Effects\n({SPEC_SHORT_NAME})',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / 'figure_21_ed_components_combined.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'figure_21_ed_components_combined.pdf', bbox_inches='tight')
    print(f"  Saved to {output_dir / 'figure_21_ed_components_combined.png'}")
    plt.show()


# %%
# =============================================================================
# COMPONENT ATTRIBUTION SUMMARY STATISTICS
# =============================================================================
if has_component_data:
    print("\n" + "="*80)
    print("EXCESS DEMAND COMPONENT ATTRIBUTION SUMMARY")
    print("="*80)

    # Filter to COVID period
    covid_mask_comp = component_decomp['period'] >= '2020-01-01'

    print(f"\nCumulative contribution to SHORTAGES (2020+):")
    print("-"*60)
    wage_contr_short = component_decomp.loc[covid_mask_comp, 'wage_contr_shortage'].sum()
    cu_contr_short = component_decomp.loc[covid_mask_comp, 'cu_contr_shortage'].sum()
    ngdppot_contr_short = component_decomp.loc[covid_mask_comp, 'ngdppot_contr_shortage'].sum()
    total_ed_short = component_decomp.loc[covid_mask_comp, 'ed_contr_shortage'].sum()

    print(f"  Wages:          {wage_contr_short:7.2f} index points")
    print(f"  Capacity Util:  {cu_contr_short:7.2f} index points")
    print(f"  Potential GDP:  {ngdppot_contr_short:7.2f} index points")
    print(f"  ─────────────────────────────────")
    print(f"  Total ED:       {total_ed_short:7.2f} index points")

    print(f"\nCumulative contribution to INFLATION (2020+, via excess demand → shortage):")
    print("-"*60)
    wage_contr_gcpi = component_decomp.loc[covid_mask_comp, 'wage_contr_gcpi_via_ed'].sum()
    cu_contr_gcpi = component_decomp.loc[covid_mask_comp, 'cu_contr_gcpi_via_ed'].sum()
    ngdppot_contr_gcpi = component_decomp.loc[covid_mask_comp, 'ngdppot_contr_gcpi_via_ed'].sum()
    total_ed_gcpi = component_decomp.loc[covid_mask_comp, 'ed_contr_gcpi'].sum()

    print(f"  Wages:          {wage_contr_gcpi:7.2f} percentage points")
    print(f"  Capacity Util:  {cu_contr_gcpi:7.2f} percentage points")
    print(f"  Potential GDP:  {ngdppot_contr_gcpi:7.2f} percentage points")
    print(f"  ─────────────────────────────────")
    print(f"  Total ED:       {total_ed_gcpi:7.2f} percentage points")

    print(f"\nCapacity utilization TOTAL effect on inflation (2020+):")
    print("-"*60)
    cu_direct = component_decomp.loc[covid_mask_comp, 'cu_contr_gcpi_direct'].sum()
    cu_indirect = component_decomp.loc[covid_mask_comp, 'cu_contr_gcpi_via_ed'].sum()
    cu_total = component_decomp.loc[covid_mask_comp, 'cu_total_contr_gcpi'].sum()

    print(f"  Direct (via wages):       {cu_direct:7.2f} percentage points")
    print(f"  Indirect (via shortages): {cu_indirect:7.2f} percentage points")
    print(f"  ─────────────────────────────────")
    print(f"  Total CU effect:          {cu_total:7.2f} percentage points")


# %%
# =============================================================================
# FIGURE 22: BB ORIGINAL WAGE DECOMPOSITION (Replication of Figure 13 from plot_decomp.py)
# =============================================================================
print("\nCreating Figure 22: Original BB Wage Decomposition...")

# Load original BB decomposition data from the Python output folder (has period columns)
bb_decomp_dir = BASE_DIR / "(3) Core Results/Decompositions/Deprecated/Output Data Python"

try:
    bb_remove_all = pd.read_excel(bb_decomp_dir / 'remove_all.xlsx')
    bb_remove_grpe = pd.read_excel(bb_decomp_dir / 'remove_grpe.xlsx')
    bb_remove_grpf = pd.read_excel(bb_decomp_dir / 'remove_grpf.xlsx')
    bb_remove_vu = pd.read_excel(bb_decomp_dir / 'remove_vu.xlsx')
    bb_remove_short = pd.read_excel(bb_decomp_dir / 'remove_shortage.xlsx')
    bb_remove_magpty = pd.read_excel(bb_decomp_dir / 'remove_magpty.xlsx')
    bb_remove_q2 = pd.read_excel(bb_decomp_dir / 'remove_2020q2.xlsx')
    bb_remove_q3 = pd.read_excel(bb_decomp_dir / 'remove_2020q3.xlsx')
    has_bb_data = True
    print(f"  Loaded BB decomposition data: {len(bb_remove_all)} observations")
except FileNotFoundError as e:
    print(f"  WARNING: BB decomposition data not found: {e}")
    print(f"  Run decomp.py first to generate the data.")
    has_bb_data = False

if has_bb_data:
    # Convert period to datetime if needed
    if not pd.api.types.is_datetime64_any_dtype(bb_remove_all['period']):
        bb_remove_all['period'] = pd.to_datetime(bb_remove_all['period'])

    # Filter all dataframes to period >= 2019 Q3 (matching plot_decomp.py)
    bb_remove_all = bb_remove_all[bb_remove_all['period'] >= filter_date].copy().reset_index(drop=True)
    bb_remove_grpe = bb_remove_grpe[bb_remove_grpe['period'] >= filter_date].copy().reset_index(drop=True)
    bb_remove_grpf = bb_remove_grpf[bb_remove_grpf['period'] >= filter_date].copy().reset_index(drop=True)
    bb_remove_vu = bb_remove_vu[bb_remove_vu['period'] >= filter_date].copy().reset_index(drop=True)
    bb_remove_short = bb_remove_short[bb_remove_short['period'] >= filter_date].copy().reset_index(drop=True)
    bb_remove_magpty = bb_remove_magpty[bb_remove_magpty['period'] >= filter_date].copy().reset_index(drop=True)
    bb_remove_q2 = bb_remove_q2[bb_remove_q2['period'] >= filter_date].copy().reset_index(drop=True)
    bb_remove_q3 = bb_remove_q3[bb_remove_q3['period'] >= filter_date].copy().reset_index(drop=True)

    print(f"  Filtered to {len(bb_remove_all)} observations from {filter_date}")

    # Create decomposition data for wages (matching plot_decomp.py Figure 13)
    bb_decomp_gw = pd.DataFrame({
        'period': bb_remove_all['period'],
        'Initial Conditions': bb_remove_all['gw_simul'],
        'Energy Prices': bb_remove_grpe['grpe_contr_gw'],
        'Food Prices': bb_remove_grpf['grpf_contr_gw'],
        'Shortages': bb_remove_short['shortage_contr_gw'],
        'V/U': bb_remove_vu['vu_contr_gw'],
        'Productivity': bb_remove_magpty['magpty_contr_gw'],
        'Q2 Dummy': bb_remove_q2['dummy2020_q2_contr_gw'],
        'Q3 Dummy': bb_remove_q3['dummy2020_q3_contr_gw']
    })

    bb_actual_gw = bb_remove_all['gw'].values
    bb_quarter_labels = [period_to_quarter_label(p) for p in bb_remove_all['period']]

    # Stack order for BB (no Capacity Util) - same as plot_decomp.py
    stack_order_bb_wage = ['Initial Conditions', 'Q3 Dummy', 'Q2 Dummy', 'Productivity',
                          'V/U', 'Food Prices', 'Energy Prices', 'Shortages']

    fig, ax = plt.subplots(figsize=(14, 8))

    x_bb = np.arange(len(bb_decomp_gw))
    width = 0.6

    bottom_pos = np.zeros(len(bb_decomp_gw))
    bottom_neg = np.zeros(len(bb_decomp_gw))

    for component in stack_order_bb_wage:
        values = bb_decomp_gw[component].values
        pos_vals = np.where(values >= 0, values, 0)
        neg_vals = np.where(values < 0, values, 0)

        ax.bar(x_bb, pos_vals, width, bottom=bottom_pos, label=component,
               color=colors[component], edgecolor='white', linewidth=0.5)
        bottom_pos += pos_vals

        ax.bar(x_bb, neg_vals, width, bottom=bottom_neg,
               color=colors[component], edgecolor='white', linewidth=0.5)
        bottom_neg += neg_vals

    ax.plot(x_bb, bb_actual_gw, color='black', linewidth=2, label='Actual Wage Inflation', marker='', zorder=10)

    ax.set_title('Original sources of wage inflation (BB)',
                 fontsize=17.5, fontweight='normal')
    ax.set_xlabel('Quarter', fontsize=16)
    ax.set_ylabel('Percent', fontsize=16)

    # X-axis ticks - every other quarter, with 45 degree rotation (matching plot_decomp.py)
    bb_tick_positions = x_bb[::2]
    bb_tick_labels_subset = [bb_quarter_labels[i] for i in bb_tick_positions]
    ax.set_xticks(bb_tick_positions)
    ax.set_xticklabels(bb_tick_labels_subset, rotation=0, ha='right')

    # Y-axis (matching plot_decomp.py)
    ax.set_ylim(-6, 8)
    ax.set_yticks(np.arange(-5, 8, 2))

    ax.yaxis.grid(True, linestyle='-', linewidth=0.5, color='lightgray', zorder=0)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.axhline(y=0, color='black', linewidth=0.5, zorder=1)

    legend_order_bb = list(reversed(stack_order_bb_wage))
    handles = [mpatches.Patch(color=colors[comp], label=comp) for comp in legend_order_bb]
    handles.append(plt.Line2D([0], [0], color='black', linewidth=2, label='Actual Wage Inflation'))
    ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, -0.1),
              ncol=5, frameon=True, edgecolor='black', fancybox=False)

    plt.tight_layout()
    plt.savefig(output_dir / 'figure_22_bb_wage_decomposition.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'figure_22_bb_wage_decomposition.pdf', bbox_inches='tight')
    print(f"  Saved to {output_dir / 'figure_22_bb_wage_decomposition.png'}")
    plt.show()


# %%
print("\n" + "="*80)
print("PLOTTING COMPLETE!")
print("="*80)
print(f"\nOutput files saved to: {output_dir}")
print("  - figure_12_price_decomposition_new.png/pdf")
print("  - figure_13_wage_decomposition_new.png/pdf")
print("  - figure_14_shortage_decomposition.png/pdf")
print("  - figure_15_alt_inflation_decomposition.png/pdf  [NEW: Excess Demand + GSCPI + CapUtil]")
print("  - figure_16_comparison_decomposition.png/pdf     [NEW: Side-by-side comparison]")
print("  - figure_17_combined_decomposition.png/pdf")
print("  - figure_18_ed_components_shortage.png/pdf       [NEW: ED components → Shortages]")
print("  - figure_19_ed_components_inflation.png/pdf      [NEW: ED components → Inflation]")
print("  - figure_20_cu_total_inflation.png/pdf           [NEW: CU total effect on inflation]")
print("  - figure_21_ed_components_combined.png/pdf       [NEW: Combined panel]")
print("  - figure_22_bb_wage_decomposition.png/pdf        [BB: Original wage decomposition]")
print("\n")

# %%
