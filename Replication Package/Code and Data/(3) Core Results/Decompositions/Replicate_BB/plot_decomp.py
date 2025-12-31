# %%
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
What Caused the U.S Pandemic-Era Inflation? Bernanke, Blanchard 2023
Python replication of plot_decomp.R
This version: December 3, 2024

This file generates plots for the decompositions in Bernanke and Blanchard (2023).

Generates:
- Figure 12: The Sources of Price Inflation, 2020 Q1 to 2023 Q2
- Figure 13: The Sources of Wage Inflation, 2020 Q1 to 2023 Q2
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# %%

#****************************CHANGE PATH HERE************************************
# Input Location - output from decomp.py
input_dir = Path("/Users/johnathansun/Documents/ec1499/Replication Package/Code and Data/(3) Core Results/Decompositions/Deprecated/Output Data Python")

# Output Location for figures
output_dir = Path("/Users/johnathansun/Documents/ec1499/Replication Package/Code and Data/(3) Core Results/Decompositions/Deprecated/Output Data Python")
output_dir.mkdir(parents=True, exist_ok=True)

#*********************************************************************************

print("Loading decomposition results...")

# Load decomposition data
remove_all = pd.read_excel(input_dir / 'remove_all.xlsx')
remove_grpe = pd.read_excel(input_dir / 'remove_grpe.xlsx')
remove_grpf = pd.read_excel(input_dir / 'remove_grpf.xlsx')
remove_vu = pd.read_excel(input_dir / 'remove_vu.xlsx')
remove_short = pd.read_excel(input_dir / 'remove_shortage.xlsx')
remove_magpty = pd.read_excel(input_dir / 'remove_magpty.xlsx')
remove_q2 = pd.read_excel(input_dir / 'remove_2020q2.xlsx')
remove_q3 = pd.read_excel(input_dir / 'remove_2020q3.xlsx')

print(f"Loaded {len(remove_all)} observations")

# %%
# Convert period to datetime if needed
if not pd.api.types.is_datetime64_any_dtype(remove_all['period']):
    remove_all['period'] = pd.to_datetime(remove_all['period'])

# Filter to period >= 2019 Q3 (July 2019)
filter_date = '2019-07-01'
remove_all = remove_all[remove_all['period'] >= filter_date].copy().reset_index(drop=True)
remove_grpe = remove_grpe[remove_grpe['period'] >= filter_date].copy().reset_index(drop=True)
remove_grpf = remove_grpf[remove_grpf['period'] >= filter_date].copy().reset_index(drop=True)
remove_vu = remove_vu[remove_vu['period'] >= filter_date].copy().reset_index(drop=True)
remove_short = remove_short[remove_short['period'] >= filter_date].copy().reset_index(drop=True)
remove_magpty = remove_magpty[remove_magpty['period'] >= filter_date].copy().reset_index(drop=True)
remove_q2 = remove_q2[remove_q2['period'] >= filter_date].copy().reset_index(drop=True)
remove_q3 = remove_q3[remove_q3['period'] >= filter_date].copy().reset_index(drop=True)

print(f"Filtered to {len(remove_all)} observations from {filter_date}")

# %%
# Helper function to format quarter labels
def period_to_quarter_label(dt):
    """Convert datetime to 'YYYY QN' format"""
    if pd.isna(dt):
        return ''
    return f"{dt.year} Q{(dt.month - 1) // 3 + 1}"

# Create quarter labels
quarter_labels = [period_to_quarter_label(p) for p in remove_all['period']]

# %%
# Define colors matching R script
colors = {
    'Initial Conditions': 'grey',
    'V/U': 'red',
    'Energy Prices': 'blue',
    'Food Prices': 'skyblue',
    'Shortages': 'gold',
    'Productivity': 'orange',
    'Q2 Dummy': 'darkgreen',
    'Q3 Dummy': 'lightgreen'
}

# Order for stacking (bottom to top) - Initial Conditions at bottom
stack_order = ['Initial Conditions', 'Q3 Dummy', 'Q2 Dummy', 'Productivity',
               'V/U', 'Food Prices', 'Energy Prices', 'Shortages']

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
# Figure 12: THE SOURCES OF PRICE INFLATION
print("\nCreating Figure 12: Sources of Price Inflation...")

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

actual_gcpi = remove_all['gcpi'].values

fig, ax = plt.subplots(figsize=(14, 8))

# Create x positions
x = np.arange(len(decomp_gcpi))
width = 0.6

# Stack bars
bottom_pos = np.zeros(len(decomp_gcpi))
bottom_neg = np.zeros(len(decomp_gcpi))

for component in stack_order:
    values = decomp_gcpi[component].values
    # Separate positive and negative values for proper stacking
    pos_vals = np.where(values >= 0, values, 0)
    neg_vals = np.where(values < 0, values, 0)

    # Stack positive values
    ax.bar(x, pos_vals, width, bottom=bottom_pos, label=component,
           color=colors[component], edgecolor='white', linewidth=0.5)
    bottom_pos += pos_vals

    # Stack negative values
    ax.bar(x, neg_vals, width, bottom=bottom_neg,
           color=colors[component], edgecolor='white', linewidth=0.5)
    bottom_neg += neg_vals

# Add actual inflation line
ax.plot(x, actual_gcpi, color='black', linewidth=2, label='Actual Inflation', marker='', zorder=10)

ax.set_title('Figure 12. THE SOURCES OF PRICE INFLATION, 2020 Q1 to 2023 Q2',
             fontsize=17.5, fontweight='normal')
ax.set_xlabel('Quarter', fontsize=16)
ax.set_ylabel('Percent', fontsize=16)

# Set x-axis ticks
tick_positions = x[::2]  # Every other quarter
tick_labels = [quarter_labels[i] for i in tick_positions]
ax.set_xticks(tick_positions)
ax.set_xticklabels(tick_labels, rotation=45, ha='right')

# Set y-axis
ax.set_ylim(-4, 12)
ax.set_yticks(np.arange(-4, 13, 2))

# Gridlines
ax.yaxis.grid(True, linestyle='-', linewidth=0.5, color='lightgray', zorder=0)
ax.xaxis.grid(False)
ax.set_axisbelow(True)

# Remove top and right spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Add horizontal line at y=0
ax.axhline(y=0, color='black', linewidth=0.5, zorder=1)

# Create legend (reverse order so it reads top-to-bottom matching visual)
legend_order = list(reversed(stack_order))
handles = [mpatches.Patch(color=colors[comp], label=comp) for comp in legend_order]
handles.append(plt.Line2D([0], [0], color='black', linewidth=2, label='Actual Inflation'))
ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, -0.15),
          ncol=5, frameon=True, edgecolor='black', fancybox=False)

plt.tight_layout()
plt.savefig(output_dir / 'figure_12_price_decomposition.png', dpi=300, bbox_inches='tight')
plt.savefig(output_dir / 'figure_12_price_decomposition.pdf', bbox_inches='tight')
print(f"  Saved to {output_dir / 'figure_12_price_decomposition.png'}")
plt.show()

# %%
# Figure 13: THE SOURCES OF WAGE INFLATION
print("\nCreating Figure 13: Sources of Wage Inflation...")

# Create decomposition data for wages
decomp_gw = pd.DataFrame({
    'period': remove_all['period'],
    'Initial Conditions': remove_all['gw_simul'],
    'Energy Prices': remove_grpe['grpe_contr_gw'],
    'Food Prices': remove_grpf['grpf_contr_gw'],
    'Shortages': remove_short['shortage_contr_gw'],
    'V/U': remove_vu['vu_contr_gw'],
    'Productivity': remove_magpty['magpty_contr_gw'],
    'Q2 Dummy': remove_q2['dummy2020_q2_contr_gw'],
    'Q3 Dummy': remove_q3['dummy2020_q3_contr_gw']
})

actual_gw = remove_all['gw'].values

fig, ax = plt.subplots(figsize=(14, 8))

# Stack bars
bottom_pos = np.zeros(len(decomp_gw))
bottom_neg = np.zeros(len(decomp_gw))

for component in stack_order:
    values = decomp_gw[component].values
    pos_vals = np.where(values >= 0, values, 0)
    neg_vals = np.where(values < 0, values, 0)

    ax.bar(x, pos_vals, width, bottom=bottom_pos, label=component,
           color=colors[component], edgecolor='white', linewidth=0.5)
    bottom_pos += pos_vals

    ax.bar(x, neg_vals, width, bottom=bottom_neg,
           color=colors[component], edgecolor='white', linewidth=0.5)
    bottom_neg += neg_vals

# Add actual wage inflation line
ax.plot(x, actual_gw, color='black', linewidth=2, label='Actual Wage Inflation', marker='', zorder=10)

ax.set_title('Figure 13. THE SOURCES OF WAGE INFLATION, 2020 Q1 to 2023 Q2',
             fontsize=17.5, fontweight='normal')
ax.set_xlabel('Quarter', fontsize=16)
ax.set_ylabel('Percent', fontsize=16)

# Set x-axis ticks
ax.set_xticks(tick_positions)
ax.set_xticklabels(tick_labels, rotation=45, ha='right')

# Set y-axis
ax.set_ylim(-4, 10)
ax.set_yticks(np.arange(-4, 11, 2))

# Gridlines
ax.yaxis.grid(True, linestyle='-', linewidth=0.5, color='lightgray', zorder=0)
ax.xaxis.grid(False)
ax.set_axisbelow(True)

# Remove top and right spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Add horizontal line at y=0
ax.axhline(y=0, color='black', linewidth=0.5, zorder=1)

# Create legend (reverse order so it reads top-to-bottom matching visual)
handles = [mpatches.Patch(color=colors[comp], label=comp) for comp in legend_order]
handles.append(plt.Line2D([0], [0], color='black', linewidth=2, label='Actual Wage Inflation'))
ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, -0.15),
          ncol=5, frameon=True, edgecolor='black', fancybox=False)

plt.tight_layout()
plt.savefig(output_dir / 'figure_13_wage_decomposition.png', dpi=300, bbox_inches='tight')
plt.savefig(output_dir / 'figure_13_wage_decomposition.pdf', bbox_inches='tight')
print(f"  Saved to {output_dir / 'figure_13_wage_decomposition.png'}")
plt.show()

# %%
# Create combined figure
print("\nCreating combined figure...")

fig, axes = plt.subplots(1, 2, figsize=(18, 7))

# Figure 12 (left)
ax1 = axes[0]
bottom_pos = np.zeros(len(decomp_gcpi))
bottom_neg = np.zeros(len(decomp_gcpi))

for component in stack_order:
    values = decomp_gcpi[component].values
    pos_vals = np.where(values >= 0, values, 0)
    neg_vals = np.where(values < 0, values, 0)
    ax1.bar(x, pos_vals, width, bottom=bottom_pos, color=colors[component], edgecolor='white', linewidth=0.3)
    bottom_pos += pos_vals
    ax1.bar(x, neg_vals, width, bottom=bottom_neg, color=colors[component], edgecolor='white', linewidth=0.3)
    bottom_neg += neg_vals

ax1.plot(x, actual_gcpi, color='black', linewidth=2, zorder=10)
ax1.set_title('Figure 12. Price Inflation Decomposition', fontsize=14)
ax1.set_xlabel('Quarter', fontsize=12)
ax1.set_ylabel('Percent', fontsize=12)
ax1.set_xticks(tick_positions)
ax1.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=10)
ax1.set_ylim(-4, 12)
ax1.yaxis.grid(True, linestyle='-', linewidth=0.5, color='lightgray', zorder=0)
ax1.set_axisbelow(True)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.axhline(y=0, color='black', linewidth=0.5, zorder=1)

# Figure 13 (right)
ax2 = axes[1]
bottom_pos = np.zeros(len(decomp_gw))
bottom_neg = np.zeros(len(decomp_gw))

for component in stack_order:
    values = decomp_gw[component].values
    pos_vals = np.where(values >= 0, values, 0)
    neg_vals = np.where(values < 0, values, 0)
    ax2.bar(x, pos_vals, width, bottom=bottom_pos, color=colors[component], edgecolor='white', linewidth=0.3)
    bottom_pos += pos_vals
    ax2.bar(x, neg_vals, width, bottom=bottom_neg, color=colors[component], edgecolor='white', linewidth=0.3)
    bottom_neg += neg_vals

ax2.plot(x, actual_gw, color='black', linewidth=2, zorder=10)
ax2.set_title('Figure 13. Wage Inflation Decomposition', fontsize=14)
ax2.set_xlabel('Quarter', fontsize=12)
ax2.set_ylabel('Percent', fontsize=12)
ax2.set_xticks(tick_positions)
ax2.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=10)
ax2.set_ylim(-4, 10)
ax2.yaxis.grid(True, linestyle='-', linewidth=0.5, color='lightgray', zorder=0)
ax2.set_axisbelow(True)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.axhline(y=0, color='black', linewidth=0.5, zorder=1)

# Create shared legend (reverse order so it reads top-to-bottom matching visual)
handles = [mpatches.Patch(color=colors[comp], label=comp) for comp in legend_order]
handles.append(plt.Line2D([0], [0], color='black', linewidth=2, label='Actual'))
fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 0.02),
           ncol=5, frameon=True, edgecolor='black', fancybox=False, fontsize=10)

plt.suptitle('Decomposition of Inflation: Price and Wage Components',
             fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(output_dir / 'figures_12_13_combined.png', dpi=300, bbox_inches='tight')
plt.savefig(output_dir / 'figures_12_13_combined.pdf', bbox_inches='tight')
print(f"  Saved to {output_dir / 'figures_12_13_combined.png'}")
plt.show()

# %%
# Print summary statistics
print("\n" + "="*80)
print("DECOMPOSITION SUMMARY")
print("="*80)

# Find peak contributions
print(f"\nPeak contributions to PRICE inflation (gcpi):")
print(f"  Energy Prices:      {decomp_gcpi['Energy Prices'].max():.2f} (Q: {quarter_labels[decomp_gcpi['Energy Prices'].argmax()]})")
print(f"  Food Prices:        {decomp_gcpi['Food Prices'].max():.2f} (Q: {quarter_labels[decomp_gcpi['Food Prices'].argmax()]})")
print(f"  Shortages:          {decomp_gcpi['Shortages'].max():.2f} (Q: {quarter_labels[decomp_gcpi['Shortages'].argmax()]})")
print(f"  V/U:                {decomp_gcpi['V/U'].max():.2f} (Q: {quarter_labels[decomp_gcpi['V/U'].argmax()]})")
print(f"  Productivity:       {decomp_gcpi['Productivity'].max():.2f} (Q: {quarter_labels[decomp_gcpi['Productivity'].argmax()]})")

print(f"\nPeak contributions to WAGE inflation (gw):")
print(f"  Energy Prices:      {decomp_gw['Energy Prices'].max():.2f} (Q: {quarter_labels[decomp_gw['Energy Prices'].argmax()]})")
print(f"  Food Prices:        {decomp_gw['Food Prices'].max():.2f} (Q: {quarter_labels[decomp_gw['Food Prices'].argmax()]})")
print(f"  Shortages:          {decomp_gw['Shortages'].max():.2f} (Q: {quarter_labels[decomp_gw['Shortages'].argmax()]})")
print(f"  V/U:                {decomp_gw['V/U'].max():.2f} (Q: {quarter_labels[decomp_gw['V/U'].argmax()]})")
print(f"  Productivity:       {decomp_gw['Productivity'].max():.2f} (Q: {quarter_labels[decomp_gw['Productivity'].argmax()]})")

print("\n" + "="*80)
print("PLOTTING COMPLETE!")
print("="*80)
print(f"\nOutput files saved to: {output_dir}")
print("  - figure_12_price_decomposition.png/pdf")
print("  - figure_13_wage_decomposition.png/pdf")
print("  - figures_12_13_combined.png/pdf")
print("\n")

# %%
