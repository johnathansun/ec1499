# %%
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modified Bernanke-Blanchard Model - Predicted vs. Actual Plots
Liang & Sun (2025)

This file generates plots of predicted versus actual values for the new model.
The wage and inflation expectations equations are estimated on the pre-COVID sample
and an out-of-sample prediction is performed. Price and shortage equations use full sample.

Generates:
- Figure 3: Wage Growth (gw vs gwf1) - now with capacity utilization
- Figure 7: Inflation (gcpi vs gcpif)
- Figure 8: Short-run Inflation Expectations (cf1 vs cf1f)
- Figure 9: Long-run Inflation Expectations (cf10 vs cf10f)
- Figure NEW: Shortage (shortage vs shortagef) - NEW equation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

# %%

#****************************CHANGE PATH HERE************************************
# Input Location - output from regression_new_model_pre_covid.py
input_path = Path("/Users/johnathansun/Documents/ec1499/Replication Package/Code and Data/(2) Regressions/Output Data Python (New Model Pre Covid)/eq_simulations_data_new_model_pre_covid.xlsx")

# Output Location for figures
output_dir = Path("/Users/johnathansun/Documents/ec1499/Replication Package/Code and Data/(2) Regressions/Predicted vs. Actual/Figures Python (New Model)")
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

# Quarter formatter for x-axis
def quarter_formatter(x, pos):
    dt = mdates.num2date(x)
    q = (dt.month - 1) // 3 + 1
    return f"{dt.year} Q{q}"

# %%
# Figure 3: WAGE GROWTH (Modified with Capacity Utilization)
print("\nCreating Figure 3: Wage Growth (Modified Model)...")

fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(df['period'], df['gw'], color='darkblue', linewidth=1.5, label='Actual')
ax.plot(df['period'], df['gwf1'], color='darkred', linewidth=1.5, label='Predicted')

ax.set_title('Figure 3. WAGE GROWTH (New Model), 2020 Q1 - 2023 Q2.', fontsize=17.5, fontweight='normal')
ax.set_ylabel('Percent', fontsize=16)
ax.set_xlabel('')

# Format x-axis - show every 2 quarters (Q1 and Q3)
ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7]))
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
plt.savefig(output_dir / 'figure_3_wage_growth_new_model.png', dpi=300, bbox_inches='tight')
plt.savefig(output_dir / 'figure_3_wage_growth_new_model.pdf', bbox_inches='tight')
print(f"  Saved to {output_dir / 'figure_3_wage_growth_new_model.png'}")
plt.show()

# %%
# Figure NEW: SHORTAGE (New Equation)
print("\nCreating Figure: Shortage (New Equation)...")

fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(df['period'], df['shortage'], color='darkblue', linewidth=1.5, label='Actual')
ax.plot(df['period'], df['shortagef'], color='darkred', linewidth=1.5, label='Predicted')

ax.set_title('SHORTAGE, 2020 Q1 - 2023 Q2.', fontsize=17.5, fontweight='normal')
ax.set_ylabel('Index', fontsize=16)
ax.set_xlabel('')

# Format x-axis - show every 2 quarters (Q1 and Q3)
ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7]))
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
plt.savefig(output_dir / 'figure_shortage_new_model.png', dpi=300, bbox_inches='tight')
plt.savefig(output_dir / 'figure_shortage_new_model.pdf', bbox_inches='tight')
print(f"  Saved to {output_dir / 'figure_shortage_new_model.png'}")
plt.show()

# %%
# Figure 7: INFLATION
print("\nCreating Figure 7: Inflation...")

fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(df['period'], df['gcpi'], color='darkblue', linewidth=1.5, label='Actual')
ax.plot(df['period'], df['gcpif'], color='darkred', linewidth=1.5, label='Predicted')

ax.set_title('Figure 7. INFLATION (New Model), 2020 Q1 - 2023 Q2.', fontsize=17.5, fontweight='normal')
ax.set_ylabel('Percent', fontsize=16)
ax.set_xlabel('')

# Set y-axis limits
ax.set_ylim(-2, 12)
ax.set_yticks([-2, 0, 2, 4, 6, 8, 10, 12])

# Format x-axis - show every 2 quarters (Q1 and Q3)
ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7]))
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
plt.savefig(output_dir / 'figure_7_inflation_new_model.png', dpi=300, bbox_inches='tight')
plt.savefig(output_dir / 'figure_7_inflation_new_model.pdf', bbox_inches='tight')
print(f"  Saved to {output_dir / 'figure_7_inflation_new_model.png'}")
plt.show()

# %%
# Figure 8: SHORT-RUN INFLATION EXPECTATIONS
print("\nCreating Figure 8: Short-run Inflation Expectations...")

fig, ax = plt.subplots(figsize=(8, 6))

df_plot = df[df['period'] <= '2023-01-01'].copy()

ax.plot(df_plot['period'], df_plot['cf1'], color='darkblue', linewidth=1.5, label='Actual')
ax.plot(df_plot['period'], df_plot['cf1f'], color='darkred', linewidth=1.5, label='Predicted')

ax.set_title('SHORT-RUN EXPECTATIONS, 2020 Q1 - 2023 Q1.', fontsize=17.5, fontweight='normal')
ax.set_ylabel('Percent', fontsize=16)
ax.set_xlabel('')

# Format x-axis - show every 2 quarters (Q1 and Q3)
ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7]))
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
plt.savefig(output_dir / 'figure_8_short_run_expectations_new_model.png', dpi=300, bbox_inches='tight')
plt.savefig(output_dir / 'figure_8_short_run_expectations_new_model.pdf', bbox_inches='tight')
print(f"  Saved to {output_dir / 'figure_8_short_run_expectations_new_model.png'}")
plt.show()

# %%
# Figure 9: LONG-RUN INFLATION EXPECTATIONS
print("\nCreating Figure 9: Long-run Inflation Expectations...")

fig, ax = plt.subplots(figsize=(8, 6))

ax.plot(df_plot['period'], df_plot['cf10'], color='darkblue', linewidth=1.5, label='Actual')
ax.plot(df_plot['period'], df_plot['cf10f'], color='darkred', linewidth=1.5, label='Predicted')

ax.set_title('LONG-RUN EXPECTATIONS, 2020 Q1 - 2023 Q1.', fontsize=17.5, fontweight='normal')
ax.set_ylabel('Percent', fontsize=16)
ax.set_xlabel('')

# Set y-axis limits
ax.set_ylim(1, 2.5)
ax.set_yticks([1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4])

# Format x-axis - show every 2 quarters (Q1 and Q3)
ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7]))
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
plt.savefig(output_dir / 'figure_9_long_run_expectations_new_model.png', dpi=300, bbox_inches='tight')
plt.savefig(output_dir / 'figure_9_long_run_expectations_new_model.pdf', bbox_inches='tight')
print(f"  Saved to {output_dir / 'figure_9_long_run_expectations_new_model.png'}")
plt.show()

# %%
# Create a combined figure with all 5 plots (2x3 grid)
print("\nCreating combined figure (5 panels)...")

fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# Flatten axes for easier indexing
axes_flat = axes.flatten()

# Plot configurations: (ax, actual_col, pred_col, title, ylim, ylabel)
plots_config = [
    (axes_flat[0], 'gw', 'gwf1', 'Figure 3. WAGE GROWTH\n(with capacity utilization)', None, 'Percent'),
    (axes_flat[1], 'shortage', 'shortagef', 'NEW: SHORTAGE\n(endogenous equation)', None, 'Index'),
    (axes_flat[2], 'gcpi', 'gcpif', 'Figure 7. INFLATION', (-2, 12), 'Percent'),
    (axes_flat[3], 'cf1', 'cf1f', 'Figure 8. SHORT-RUN\nEXPECTATIONS', None, 'Percent'),
    (axes_flat[4], 'cf10', 'cf10f', 'Figure 9. LONG-RUN\nEXPECTATIONS', (1, 2.5), 'Percent'),
]

for ax, actual_col, pred_col, title, ylim, ylabel in plots_config:
    ax.plot(df['period'], df[actual_col], color='darkblue', linewidth=1.5, label='Actual')
    ax.plot(df['period'], df[pred_col], color='darkred', linewidth=1.5, label='Predicted')
    ax.set_title(title, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=11)
    # Show every 2 quarters (Q1 and Q3)
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7]))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(quarter_formatter))
    ax.tick_params(axis='x', rotation=45)
    ax.yaxis.grid(True, linestyle='-', linewidth=0.5, color='lightgray')
    ax.xaxis.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if ylim:
        ax.set_ylim(ylim)
    ax.legend(loc='best', fontsize=9, frameon=True, edgecolor='black')

# Hide the 6th subplot
axes_flat[5].axis('off')

plt.suptitle('Modified Bernanke-Blanchard Model (Liang & Sun 2025)\nOut-of-Sample Predictions: 2020 Q1 - 2023 Q2',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(output_dir / 'figures_combined_new_model.png', dpi=300, bbox_inches='tight')
plt.savefig(output_dir / 'figures_combined_new_model.pdf', bbox_inches='tight')
print(f"  Saved to {output_dir / 'figures_combined_new_model.png'}")
plt.show()


# %%
# Calculate and display prediction errors for COVID period
print("\n" + "="*80)
print("PREDICTION ERROR ANALYSIS (Out-of-Sample: 2020 Q1 - 2023 Q2)")
print("="*80)

covid_period = df[df['period'] >= '2020-01-01'].copy()

def calc_errors(actual, predicted, name):
    """Calculate RMSE and MAE"""
    valid = ~(actual.isna() | predicted.isna())
    actual_clean = actual[valid]
    pred_clean = predicted[valid]

    rmse = np.sqrt(np.mean((actual_clean - pred_clean)**2))
    mae = np.mean(np.abs(actual_clean - pred_clean))
    mean_error = np.mean(actual_clean - pred_clean)

    print(f"\n{name}:")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE:  {mae:.4f}")
    print(f"  Mean Error (Actual - Pred): {mean_error:.4f}")
    return rmse, mae, mean_error

calc_errors(covid_period['gw'], covid_period['gwf1'], "Wage Growth (gw)")
calc_errors(covid_period['shortage'], covid_period['shortagef'], "Shortage")
calc_errors(covid_period['gcpi'], covid_period['gcpif'], "Inflation (gcpi)")
calc_errors(covid_period['cf1'], covid_period['cf1f'], "Short-run Expectations (cf1)")
calc_errors(covid_period['cf10'], covid_period['cf10f'], "Long-run Expectations (cf10)")


# %%
# Compare with BB model (if data available)
print("\n" + "="*80)
print("COMPARISON: NEW MODEL vs. ORIGINAL BB MODEL")
print("="*80)

bb_input_path = Path("/Users/johnathansun/Documents/ec1499/Replication Package/Code and Data/(2) Regressions/Output Data Python (Pre Covid Sample)/eq_simulations_data_restricted_python.xlsx")

if bb_input_path.exists():
    print("\nLoading original BB model predictions...")
    df_bb = pd.read_excel(bb_input_path)

    if not pd.api.types.is_datetime64_any_dtype(df_bb['period']):
        df_bb['period'] = pd.to_datetime(df_bb['period'])

    df_bb = df_bb[df_bb['period'] >= '2020-01-01'].copy()

    # Create comparison figure for wage growth
    if 'gwf1' in df_bb.columns:
        print("\nCreating wage growth comparison figure...")

        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(covid_period['period'], covid_period['gw'], color='darkblue',
                linewidth=2, label='Actual')
        ax.plot(df_bb['period'], df_bb['gwf1'], color='darkred',
                linewidth=1.5, linestyle='--', label='BB Model Predicted')
        ax.plot(covid_period['period'], covid_period['gwf1'], color='darkgreen',
                linewidth=1.5, label='New Model Predicted')

        ax.set_title('WAGE GROWTH: BB vs. New Model Predictions', fontsize=17.5)
        ax.set_ylabel('Percent', fontsize=16)
        # Show every 2 quarters (Q1 and Q3)
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7]))
        ax.xaxis.set_major_formatter(plt.FuncFormatter(quarter_formatter))

        ax.yaxis.grid(True, linestyle='-', linewidth=0.5, color='lightgray')
        ax.xaxis.grid(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=3,
                  frameon=True, edgecolor='black', fancybox=False)

        plt.tight_layout()
        plt.savefig(output_dir / 'comparison_wage_growth.png', dpi=300, bbox_inches='tight')
        plt.savefig(output_dir / 'comparison_wage_growth.pdf', bbox_inches='tight')
        print(f"  Saved to {output_dir / 'comparison_wage_growth.png'}")
        plt.show()

        # Compute comparison statistics
        print("\n--- Wage Growth Prediction Error Comparison ---")

        rmse_bb = np.sqrt(np.mean((df_bb['gw'] - df_bb['gwf1'])**2))
        rmse_new = np.sqrt(np.mean((covid_period['gw'] - covid_period['gwf1'])**2))

        print(f"  BB Model RMSE:  {rmse_bb:.4f}")
        print(f"  New Model RMSE: {rmse_new:.4f}")
        print(f"  Improvement:    {(rmse_bb - rmse_new) / rmse_bb * 100:.1f}%")
else:
    print("\nOriginal BB model predictions not found. Run regression_pre_covid.py first.")


# %%
print("\n" + "="*80)
print("PLOTTING COMPLETE!")
print("="*80)
print(f"\nOutput files saved to: {output_dir}")
print("  - figure_3_wage_growth_new_model.png/pdf")
print("  - figure_shortage_new_model.png/pdf (NEW)")
print("  - figure_7_inflation_new_model.png/pdf")
print("  - figure_8_short_run_expectations_new_model.png/pdf")
print("  - figure_9_long_run_expectations_new_model.png/pdf")
print("  - figures_combined_new_model.png/pdf")
print("  - comparison_wage_growth.png/pdf (if BB data available)")
print("\n")

# %%
