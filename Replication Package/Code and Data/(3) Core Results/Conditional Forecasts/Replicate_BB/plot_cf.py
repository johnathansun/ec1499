# %%
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
What Caused the U.S Pandemic-Era Inflation? Bernanke, Blanchard 2023
Python replication of plot_cf.R
This version: December 3, 2024

This file generates plots of the conditional forecasts from Bernanke and Blanchard (2023).

Generates:
- Figure 14: Inflation projections for alternative paths of v/u
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# %%

#****************************CHANGE PATH HERE************************************
# Input Location - output from cond_forecast.py
input_dir = Path("/Users/johnathansun/Documents/ec1499/Replication Package/Code and Data/(3) Core Results/Conditional Forecasts/Output Data Python")

# Output Location for figures
output_dir = Path("/Users/johnathansun/Documents/ec1499/Replication Package/Code and Data/(3) Core Results/Conditional Forecasts/Figures Python")
output_dir.mkdir(parents=True, exist_ok=True)

#*********************************************************************************

print("Loading conditional forecast results...")

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
    'low': terminal_low['gcpi_simul'],
    'mid': terminal_mid['gcpi_simul'],
    'high': terminal_high['gcpi_simul']
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

# %%
# Common plot styling
def setup_plot_style():
    """Set up common plot styling similar to R ggplot2 theme_bw"""
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
# Figure 14: INFLATION PROJECTIONS FOR ALTERNATIVE PATHS OF V/U
print("\nCreating Figure 14: Conditional Inflation Forecasts...")

fig, ax = plt.subplots(figsize=(12, 7))

# Plot the three scenarios
ax.plot(cf_data['period'], cf_data['low'], color='darkblue', linewidth=1.5,
        label='v/u = 0.8')
ax.plot(cf_data['period'], cf_data['mid'], color='darkred', linewidth=1.5,
        label='v/u = 1.2 = (v/u)*')
ax.plot(cf_data['period'], cf_data['high'], color='orange', linewidth=1.5,
        label='v/u = 1.8')

ax.set_title('Figure 14. Inflation projections for alternative paths of v/u.',
             fontsize=17.5, fontweight='normal')
ax.set_xlabel('Quarter', fontsize=16)
ax.set_ylabel('Percent', fontsize=16)

# Set y-axis limits and ticks
ax.set_ylim(1.5, 4.0)
ax.set_yticks(np.arange(1.5, 4.5, 0.5))

# Format x-axis with quarterly ticks
# Show every other quarter label
x_positions = range(len(cf_data))
tick_positions = list(range(0, len(cf_data), 2))  # Every other quarter
tick_labels = [quarter_labels[i] for i in tick_positions]

ax.set_xticks([cf_data['period'].iloc[i] for i in tick_positions])
ax.set_xticklabels(tick_labels, rotation=45, ha='right')

# Gridlines - horizontal only
ax.yaxis.grid(True, linestyle='-', linewidth=0.5, color='lightgray')
ax.xaxis.grid(False)

# Remove top and right spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Legend at bottom
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3,
          frameon=True, edgecolor='black', fancybox=False)

plt.tight_layout()
plt.savefig(output_dir / 'figure_14_conditional_forecast.png', dpi=300, bbox_inches='tight')
plt.savefig(output_dir / 'figure_14_conditional_forecast.pdf', bbox_inches='tight')
print(f"  Saved to {output_dir / 'figure_14_conditional_forecast.png'}")
plt.show()

# %%
# Create additional plot showing v/u paths
print("\nCreating supplementary figure: V/U paths...")

# Get v/u data
vu_data = pd.DataFrame({
    'period': terminal_low['period'],
    'low': terminal_low['vu_simul'],
    'mid': terminal_mid['vu_simul'],
    'high': terminal_high['vu_simul']
})

# Filter to same range
vu_data = vu_data[(vu_data['period'] >= '2022-10-01') & (vu_data['period'] <= '2027-01-01')].copy()
vu_data = vu_data.reset_index(drop=True)

fig, ax = plt.subplots(figsize=(12, 7))

ax.plot(vu_data['period'], vu_data['low'], color='darkblue', linewidth=1.5,
        label='v/u -> 0.8')
ax.plot(vu_data['period'], vu_data['mid'], color='darkred', linewidth=1.5,
        label='v/u -> 1.2')
ax.plot(vu_data['period'], vu_data['high'], color='orange', linewidth=1.5,
        label='v/u -> 1.8')

ax.set_title('V/U Ratio Paths for Conditional Forecasts',
             fontsize=17.5, fontweight='normal')
ax.set_xlabel('Quarter', fontsize=16)
ax.set_ylabel('V/U Ratio', fontsize=16)

ax.set_xticks([vu_data['period'].iloc[i] for i in tick_positions])
ax.set_xticklabels(tick_labels, rotation=45, ha='right')

ax.yaxis.grid(True, linestyle='-', linewidth=0.5, color='lightgray')
ax.xaxis.grid(False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3,
          frameon=True, edgecolor='black', fancybox=False)

plt.tight_layout()
plt.savefig(output_dir / 'figure_14_vu_paths.png', dpi=300, bbox_inches='tight')
plt.savefig(output_dir / 'figure_14_vu_paths.pdf', bbox_inches='tight')
print(f"  Saved to {output_dir / 'figure_14_vu_paths.png'}")
plt.show()

# %%
# Create combined figure
print("\nCreating combined figure...")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Left: Inflation forecasts
ax1 = axes[0]
ax1.plot(cf_data['period'], cf_data['low'], color='darkblue', linewidth=1.5, label='v/u = 0.8')
ax1.plot(cf_data['period'], cf_data['mid'], color='darkred', linewidth=1.5, label='v/u = 1.2 = (v/u)*')
ax1.plot(cf_data['period'], cf_data['high'], color='orange', linewidth=1.5, label='v/u = 1.8')
ax1.set_title('Figure 14. Inflation Projections', fontsize=14)
ax1.set_xlabel('Quarter', fontsize=12)
ax1.set_ylabel('Percent', fontsize=12)
ax1.set_ylim(1.5, 4.0)
ax1.set_yticks(np.arange(1.5, 4.5, 0.5))
ax1.set_xticks([cf_data['period'].iloc[i] for i in tick_positions])
ax1.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=10)
ax1.yaxis.grid(True, linestyle='-', linewidth=0.5, color='lightgray')
ax1.xaxis.grid(False)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.legend(loc='best', fontsize=10, frameon=True, edgecolor='black')

# Right: V/U paths
ax2 = axes[1]
ax2.plot(vu_data['period'], vu_data['low'], color='darkblue', linewidth=1.5, label='v/u -> 0.8')
ax2.plot(vu_data['period'], vu_data['mid'], color='darkred', linewidth=1.5, label='v/u -> 1.2')
ax2.plot(vu_data['period'], vu_data['high'], color='orange', linewidth=1.5, label='v/u -> 1.8')
ax2.set_title('V/U Ratio Paths', fontsize=14)
ax2.set_xlabel('Quarter', fontsize=12)
ax2.set_ylabel('V/U Ratio', fontsize=12)
ax2.set_xticks([vu_data['period'].iloc[i] for i in tick_positions])
ax2.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=10)
ax2.yaxis.grid(True, linestyle='-', linewidth=0.5, color='lightgray')
ax2.xaxis.grid(False)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.legend(loc='best', fontsize=10, frameon=True, edgecolor='black')

plt.suptitle('Conditional Forecasts: Alternative V/U Scenarios',
             fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(output_dir / 'figure_14_combined.png', dpi=300, bbox_inches='tight')
plt.savefig(output_dir / 'figure_14_combined.pdf', bbox_inches='tight')
print(f"  Saved to {output_dir / 'figure_14_combined.png'}")
plt.show()

# %%
# Print summary statistics
print("\n" + "="*80)
print("CONDITIONAL FORECAST SUMMARY")
print("="*80)

print(f"\nInflation forecasts by scenario:")
print(f"\n  Scenario: v/u -> 0.8 (labor market normalizes)")
print(f"    Start (2022 Q4): {cf_data['low'].iloc[0]:.2f}%")
print(f"    End (2027 Q1):   {cf_data['low'].iloc[-1]:.2f}%")

print(f"\n  Scenario: v/u -> 1.2 (target steady state)")
print(f"    Start (2022 Q4): {cf_data['mid'].iloc[0]:.2f}%")
print(f"    End (2027 Q1):   {cf_data['mid'].iloc[-1]:.2f}%")

print(f"\n  Scenario: v/u -> 1.8 (persistent tight labor market)")
print(f"    Start (2022 Q4): {cf_data['high'].iloc[0]:.2f}%")
print(f"    End (2027 Q1):   {cf_data['high'].iloc[-1]:.2f}%")

print("\n" + "="*80)
print("PLOTTING COMPLETE!")
print("="*80)
print(f"\nOutput files saved to: {output_dir}")
print("  - figure_14_conditional_forecast.png/pdf")
print("  - figure_14_vu_paths.png/pdf")
print("  - figure_14_combined.png/pdf")
print("\n")

# %%
