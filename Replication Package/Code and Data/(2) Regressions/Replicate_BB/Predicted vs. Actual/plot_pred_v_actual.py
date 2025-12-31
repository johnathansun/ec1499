# %%
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
What Caused the U.S Pandemic-Era Inflation? Bernanke, Blanchard 2023
Python replication of plot_pred_v_actual.R
This version: December 3, 2024

This file generates plots of predicted versus actual values in the restricted equations.
Note that the wage and inflation expectations equations are estimated on the pre-COVID sample
and an out of sample prediction is performed. For the price equation, because of the shortage
variable, we estimate over the entire sample.

Generates:
- Figure 3: Wage Growth (gw vs gwf1)
- Figure 7: Inflation (gcpi vs gcpif)
- Figure 8: Short-run Inflation Expectations (cf1 vs cf1f)
- Figure 9: Long-run Inflation Expectations (cf10 vs cf10f)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

# %%

#****************************CHANGE PATH HERE************************************
# Input Location - output from regression_pre_covid.py
# input_path = Path("../Output Data (Restricted Sample)/eq_simulations_data_restricted.xls")
input_path = Path("/Users/johnathansun/Documents/ec1499/Replication Package/Code and Data/(2) Regressions/Output Data Python (Pre Covid Sample)/eq_simulations_data_restricted_python.xlsx")

# Output Location for figures
output_dir = Path("/Users/johnathansun/Documents/ec1499/Replication Package/Code and Data/(2) Regressions/Predicted vs. Actual/Figures Python")
output_dir.mkdir(parents=True, exist_ok=True)

#*********************************************************************************

print("Loading data...")
df = pd.read_excel(input_path)

# Convert period to datetime if needed
if not pd.api.types.is_datetime64_any_dtype(df['period']):
    df['period'] = pd.to_datetime(df['period'])

# Filter to period >= 2019 Q4 (December 2019)
df = df[df['period'] >= '2019-12-01'].copy()
df = df.reset_index(drop=True)

print(f"Plotting data from {df['period'].min()} to {df['period'].max()}")
print(f"Number of observations: {len(df)}")

# %%
# Helper function to format quarter labels
def period_to_quarter_label(dt):
    """Convert datetime to 'YYYY QN' format"""
    return f"{dt.year} Q{(dt.month - 1) // 3 + 1}"

# Create quarter labels for x-axis
df['quarter_label'] = df['period'].apply(period_to_quarter_label)

# %%
# Common plot styling function
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
# Figure 3: WAGE GROWTH
print("\nCreating Figure 3: Wage Growth...")

fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(df['period'], df['gw'], color='darkblue', linewidth=1.5, label='Actual')
ax.plot(df['period'], df['gwf1'], color='darkred', linewidth=1.5, label='Predicted')

ax.set_title('Figure 3. WAGE GROWTH, 2020 Q1 - 2023 Q2.', fontsize=17.5, fontweight='normal')
ax.set_ylabel('Percent', fontsize=16)
ax.set_xlabel('')

# Format x-axis with quarterly ticks
ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7]))  # Every 6 months
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y Q%q'))
# Custom formatter for quarters
def quarter_formatter(x, pos):
    dt = mdates.num2date(x)
    q = (dt.month - 1) // 3 + 1
    return f"{dt.year} Q{q}"
ax.xaxis.set_major_formatter(plt.FuncFormatter(quarter_formatter))

# Remove minor gridlines, keep only horizontal major gridlines
ax.yaxis.grid(True, linestyle='-', linewidth=0.5, color='lightgray')
ax.xaxis.grid(False)

# Remove top and right spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Legend at bottom
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=2,
          frameon=True, edgecolor='black', fancybox=False)

plt.tight_layout()
plt.savefig(output_dir / 'figure_3_wage_growth.png', dpi=300, bbox_inches='tight')
plt.savefig(output_dir / 'figure_3_wage_growth.pdf', bbox_inches='tight')
print(f"  Saved to {output_dir / 'figure_3_wage_growth.png'}")
plt.show()

# %%
# Figure 7: INFLATION
print("\nCreating Figure 7: Inflation...")

fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(df['period'], df['gcpi'], color='darkblue', linewidth=1.5, label='Actual')
ax.plot(df['period'], df['gcpif'], color='darkred', linewidth=1.5, label='Predicted')

ax.set_title('Figure 7. INFLATION, 2020 Q1 - 2023 Q2.', fontsize=17.5, fontweight='normal')
ax.set_ylabel('Percent', fontsize=16)
ax.set_xlabel('')

# Set y-axis limits
ax.set_ylim(-2, 12)
ax.set_yticks([-2, 0, 2, 4, 6, 8, 10, 12])

# Format x-axis
ax.xaxis.set_major_formatter(plt.FuncFormatter(quarter_formatter))

# Gridlines
ax.yaxis.grid(True, linestyle='-', linewidth=0.5, color='lightgray')
ax.xaxis.grid(False)

# Spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Legend
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=2,
          frameon=True, edgecolor='black', fancybox=False)

plt.tight_layout()
plt.savefig(output_dir / 'figure_7_inflation.png', dpi=300, bbox_inches='tight')
plt.savefig(output_dir / 'figure_7_inflation.pdf', bbox_inches='tight')
print(f"  Saved to {output_dir / 'figure_7_inflation.png'}")
plt.show()

# %%
# Figure 8: SHORT-RUN INFLATION EXPECTATIONS
print("\nCreating Figure 8: Short-run Inflation Expectations...")

fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(df['period'], df['cf1'], color='darkblue', linewidth=1.5, label='Actual')
ax.plot(df['period'], df['cf1f'], color='darkred', linewidth=1.5, label='Predicted')

ax.set_title('Figure 8. SHORT-RUN INFLATION EXPECTATIONS, 2020 Q1 - 2023 Q2.', fontsize=17.5, fontweight='normal')
ax.set_ylabel('Percent', fontsize=16)
ax.set_xlabel('')

# Format x-axis
ax.xaxis.set_major_formatter(plt.FuncFormatter(quarter_formatter))

# Gridlines
ax.yaxis.grid(True, linestyle='-', linewidth=0.5, color='lightgray')
ax.xaxis.grid(False)

# Spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Legend
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=2,
          frameon=True, edgecolor='black', fancybox=False)

plt.tight_layout()
plt.savefig(output_dir / 'figure_8_short_run_expectations.png', dpi=300, bbox_inches='tight')
plt.savefig(output_dir / 'figure_8_short_run_expectations.pdf', bbox_inches='tight')
print(f"  Saved to {output_dir / 'figure_8_short_run_expectations.png'}")
plt.show()

# %%
# Figure 9: LONG-RUN INFLATION EXPECTATIONS
print("\nCreating Figure 9: Long-run Inflation Expectations...")

fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(df['period'], df['cf10'], color='darkblue', linewidth=1.5, label='Actual')
ax.plot(df['period'], df['cf10f'], color='darkred', linewidth=1.5, label='Predicted')

ax.set_title('Figure 9. LONG-RUN INFLATION EXPECTATIONS, 2020 Q1 - 2023 Q2.', fontsize=17.5, fontweight='normal')
ax.set_ylabel('Percent', fontsize=16)
ax.set_xlabel('')

# Set y-axis limits
ax.set_ylim(1, 2.5)
ax.set_yticks([1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4])

# Format x-axis
ax.xaxis.set_major_formatter(plt.FuncFormatter(quarter_formatter))

# Gridlines
ax.yaxis.grid(True, linestyle='-', linewidth=0.5, color='lightgray')
ax.xaxis.grid(False)

# Spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Legend
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=2,
          frameon=True, edgecolor='black', fancybox=False)

plt.tight_layout()
plt.savefig(output_dir / 'figure_9_long_run_expectations.png', dpi=300, bbox_inches='tight')
plt.savefig(output_dir / 'figure_9_long_run_expectations.pdf', bbox_inches='tight')
print(f"  Saved to {output_dir / 'figure_9_long_run_expectations.png'}")
plt.show()

# %%
# Create a combined figure with all 4 plots (optional)
print("\nCreating combined figure...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot data
plots_config = [
    (axes[0, 0], 'gw', 'gwf1', 'Figure 3. WAGE GROWTH', None),
    (axes[0, 1], 'gcpi', 'gcpif', 'Figure 7. INFLATION', (-2, 12)),
    (axes[1, 0], 'cf1', 'cf1f', 'Figure 8. SHORT-RUN EXPECTATIONS', None),
    (axes[1, 1], 'cf10', 'cf10f', 'Figure 9. LONG-RUN EXPECTATIONS', (1, 2.5)),
]

for ax, actual_col, pred_col, title, ylim in plots_config:
    ax.plot(df['period'], df[actual_col], color='darkblue', linewidth=1.5, label='Actual')
    ax.plot(df['period'], df[pred_col], color='darkred', linewidth=1.5, label='Predicted')
    ax.set_title(title, fontsize=14)
    ax.set_ylabel('Percent', fontsize=12)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(quarter_formatter))
    ax.tick_params(axis='x', rotation=45)
    ax.yaxis.grid(True, linestyle='-', linewidth=0.5, color='lightgray')
    ax.xaxis.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if ylim:
        ax.set_ylim(ylim)
    ax.legend(loc='best', fontsize=10, frameon=True, edgecolor='black')

plt.suptitle('Predicted vs. Actual: Out-of-Sample Performance (2020 Q1 - 2023 Q2)',
             fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(output_dir / 'figures_3_7_8_9_combined.png', dpi=300, bbox_inches='tight')
plt.savefig(output_dir / 'figures_3_7_8_9_combined.pdf', bbox_inches='tight')
print(f"  Saved to {output_dir / 'figures_3_7_8_9_combined.png'}")
plt.show()

# %%
print("\n" + "="*80)
print("PLOTTING COMPLETE!")
print("="*80)
print(f"\nOutput files saved to: {output_dir}")
print("  - figure_3_wage_growth.png/pdf")
print("  - figure_7_inflation.png/pdf")
print("  - figure_8_short_run_expectations.png/pdf")
print("  - figure_9_long_run_expectations.png/pdf")
print("  - figures_3_7_8_9_combined.png/pdf")
print("\n")

# %%
