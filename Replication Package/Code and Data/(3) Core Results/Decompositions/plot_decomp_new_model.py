# %%
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modified Bernanke-Blanchard Model - Decomposition Plots
Liang & Sun (2025)

This file generates decomposition plots for the modified model.

Generates:
- Figure 12 (New): Sources of Price Inflation (with new channels)
- Figure 13 (New): Sources of Wage Inflation (with capacity utilization)
- Figure 14 (New): Shortage Decomposition (excess demand vs GSCPI)
- Figure 15 (New): Comparison with Original BB Model
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# %%

#****************************CHANGE PATH HERE************************************
# Input Location - output from decomp_new_model.py
input_dir = Path("/Users/johnathansun/Documents/ec1499/Replication Package/Code and Data/(3) Core Results/Decompositions/Output Data Python (New Model)")

# Output Location for figures
output_dir = Path("/Users/johnathansun/Documents/ec1499/Replication Package/Code and Data/(3) Core Results/Decompositions/Figures Python (New Model)")
output_dir.mkdir(parents=True, exist_ok=True)

#*********************************************************************************

print("="*80)
print("MODIFIED MODEL DECOMPOSITION PLOTS")
print("Liang & Sun (2025)")
print("="*80)

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
remove_gcu = pd.read_excel(input_dir / 'remove_gcu.xlsx')
remove_all = pd.read_excel(input_dir / 'remove_all.xlsx')

print(f"Loaded {len(baseline)} observations")

# %%
# Convert period to datetime if needed
for df in [baseline, remove_grpe, remove_grpf, remove_vu, remove_short, remove_magpty,
           remove_q2, remove_q3, remove_excess_demand, remove_gscpi, remove_gcu, remove_all]:
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
    'remove_gcu': remove_gcu,
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
remove_gcu = datasets['remove_gcu']
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
# Define colors - expanded for new model
colors = {
    'Initial Conditions': 'grey',
    'V/U': 'red',
    'Energy Prices': 'blue',
    'Food Prices': 'skyblue',
    'Shortages': 'gold',
    'Productivity': 'orange',
    'Q2 Dummy': 'darkgreen',
    'Q3 Dummy': 'lightgreen',
    # NEW colors for new model
    'Capacity Util': 'purple',
    'Excess Demand': 'crimson',
    'GSCPI': 'teal'
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

ax.set_title('Figure 12 (New Model). THE SOURCES OF PRICE INFLATION',
             fontsize=17.5, fontweight='normal')
ax.set_xlabel('Quarter', fontsize=16)
ax.set_ylabel('Percent', fontsize=16)

tick_positions = x[::2]
tick_labels_subset = [quarter_labels[i] for i in tick_positions]
ax.set_xticks(tick_positions)
ax.set_xticklabels(tick_labels_subset, rotation=45, ha='right')

ax.set_ylim(-4, 12)
ax.set_yticks(np.arange(-4, 13, 2))

ax.yaxis.grid(True, linestyle='-', linewidth=0.5, color='lightgray', zorder=0)
ax.xaxis.grid(False)
ax.set_axisbelow(True)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.axhline(y=0, color='black', linewidth=0.5, zorder=1)

legend_order = list(reversed(stack_order_price))
handles = [mpatches.Patch(color=colors[comp], label=comp) for comp in legend_order]
handles.append(plt.Line2D([0], [0], color='black', linewidth=2, label='Actual Inflation'))
ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, -0.15),
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
    'Capacity Util': remove_gcu['gcu_contr_gw'],  # NEW
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

ax.set_title('Figure 13 (New Model). THE SOURCES OF WAGE INFLATION\n(Includes Capacity Utilization)',
             fontsize=17.5, fontweight='normal')
ax.set_xlabel('Quarter', fontsize=16)
ax.set_ylabel('Percent', fontsize=16)

ax.set_xticks(tick_positions)
ax.set_xticklabels(tick_labels_subset, rotation=45, ha='right')
ax.set_ylim(-4, 10)
ax.set_yticks(np.arange(-4, 11, 2))

ax.yaxis.grid(True, linestyle='-', linewidth=0.5, color='lightgray', zorder=0)
ax.xaxis.grid(False)
ax.set_axisbelow(True)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.axhline(y=0, color='black', linewidth=0.5, zorder=1)

legend_order = list(reversed(stack_order_wage))
handles = [mpatches.Patch(color=colors[comp], label=comp) for comp in legend_order]
handles.append(plt.Line2D([0], [0], color='black', linewidth=2, label='Actual Wage Inflation'))
ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, -0.15),
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

ax.set_title('Figure 14 (New Model). SHORTAGE DECOMPOSITION\nExcess Demand vs Supply Chain Pressure',
             fontsize=17.5, fontweight='normal')
ax.set_xlabel('Quarter', fontsize=16)
ax.set_ylabel('Shortage Index', fontsize=16)

ax.set_xticks(tick_positions)
ax.set_xticklabels(tick_labels_subset, rotation=45, ha='right')

ax.yaxis.grid(True, linestyle='-', linewidth=0.5, color='lightgray', zorder=0)
ax.xaxis.grid(False)
ax.set_axisbelow(True)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

legend_order = list(reversed(stack_order_shortage))
handles = [mpatches.Patch(color=colors[comp], label=comp) for comp in legend_order]
handles.append(plt.Line2D([0], [0], color='black', linewidth=2, label='Actual Shortage'))
handles.append(plt.Line2D([0], [0], color='black', linewidth=2, linestyle='--', label='Simulated Shortage'))
ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, -0.12),
          ncol=5, frameon=True, edgecolor='black', fancybox=False)

plt.tight_layout()
plt.savefig(output_dir / 'figure_14_shortage_decomposition.png', dpi=300, bbox_inches='tight')
plt.savefig(output_dir / 'figure_14_shortage_decomposition.pdf', bbox_inches='tight')
print(f"  Saved to {output_dir / 'figure_14_shortage_decomposition.png'}")
plt.show()


# %%
# =============================================================================
# FIGURE 15 (NEW): COMBINED 3-PANEL FIGURE
# =============================================================================
print("\nCreating Figure 15: Combined 3-Panel Decomposition...")

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
ax1.set_xticklabels(tick_labels_subset, rotation=45, ha='right', fontsize=9)
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
ax2.set_xticklabels(tick_labels_subset, rotation=45, ha='right', fontsize=9)
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
ax3.set_xticklabels(tick_labels_subset, rotation=45, ha='right', fontsize=9)
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

plt.suptitle('Decomposition of Pandemic-Era Inflation (Modified Model)',
             fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(output_dir / 'figure_15_combined_decomposition.png', dpi=300, bbox_inches='tight')
plt.savefig(output_dir / 'figure_15_combined_decomposition.pdf', bbox_inches='tight')
print(f"  Saved to {output_dir / 'figure_15_combined_decomposition.png'}")
plt.show()


# %%
# =============================================================================
# SUMMARY STATISTICS
# =============================================================================
print("\n" + "="*80)
print("DECOMPOSITION SUMMARY")
print("="*80)

print(f"\nPeak contributions to PRICE inflation (gcpi):")
print("-"*60)
print(f"  Energy Prices:      {decomp_gcpi['Energy Prices'].max():.2f} (Q: {quarter_labels[decomp_gcpi['Energy Prices'].argmax()]})")
print(f"  Food Prices:        {decomp_gcpi['Food Prices'].max():.2f} (Q: {quarter_labels[decomp_gcpi['Food Prices'].argmax()]})")
print(f"  Shortages:          {decomp_gcpi['Shortages'].max():.2f} (Q: {quarter_labels[decomp_gcpi['Shortages'].argmax()]})")
print(f"  V/U:                {decomp_gcpi['V/U'].max():.2f} (Q: {quarter_labels[decomp_gcpi['V/U'].argmax()]})")
print(f"  Productivity:       {decomp_gcpi['Productivity'].max():.2f} (Q: {quarter_labels[decomp_gcpi['Productivity'].argmax()]})")

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


print("\n" + "="*80)
print("PLOTTING COMPLETE!")
print("="*80)
print(f"\nOutput files saved to: {output_dir}")
print("  - figure_12_price_decomposition_new.png/pdf")
print("  - figure_13_wage_decomposition_new.png/pdf")
print("  - figure_14_shortage_decomposition.png/pdf")
print("  - figure_15_combined_decomposition.png/pdf")
print("\n")

# %%
