# %%
"""
Predicted vs. Actual Plots: New Model
Liang & Sun (2025)

This script generates plots of predicted versus actual values for the new model equations.
Supports multiple specification variants via configuration flags.

Configuration flags at the top control which specification to plot.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
from pathlib import Path


def quarter_formatter(x, pos):
    """Format datetime tick as 'YYYY QN' (e.g., '2021 Q1')"""
    date = mdates.num2date(x)
    quarter = (date.month - 1) // 3 + 1
    return f"{date.year} Q{quarter}"

# %%
#****************************CONFIGURATION**************************************
# These flags MUST match the specification used in regression_new_model.py!

USE_PRE_COVID_SAMPLE = True        # Use pre-COVID sample estimates (for out-of-sample prediction)
USE_LOG_CU_WAGES = False           # True = log(CU), False = level CU in wage equation
USE_CONTEMP_CU = False             # True = CU lags 0-4, False = CU lags 1-4
USE_DETRENDED_EXCESS_DEMAND = True # Detrend excess demand in shortage equation

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
    if USE_DETRENDED_EXCESS_DEMAND:
        parts.append("Detrended ED")
    return ", ".join(parts)

SPEC_DIR_NAME = get_spec_dir_name()
SPEC_SHORT_NAME = get_spec_short_name()

# Input paths
NEW_MODEL_PATH = BASE_DIR / "(2) Regressions" / SPEC_DIR_NAME / (
    "eq_simulations_data_new_model_pre_covid.xlsx" if USE_PRE_COVID_SAMPLE
    else "eq_simulations_data_new_model.xlsx"
)
BB_MODEL_PATH = BASE_DIR / "(2) Regressions/Old Output/Output Data (Pre Covid Sample)/eq_simulations_data_restricted.xls"

# Output path
OUTPUT_DIR = BASE_DIR / "(2) Regressions/Predicted vs. Actual/Figures Python (New Model)" / SPEC_DIR_NAME
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

#*********************************************************************************

print("="*80)
print("PREDICTED VS. ACTUAL PLOTS - NEW MODEL")
print("Liang & Sun (2025)")
print("="*80)

print(f"\nSpecification: {SPEC_SHORT_NAME}")
print(f"  USE_PRE_COVID_SAMPLE:        {USE_PRE_COVID_SAMPLE}")
print(f"  USE_LOG_CU_WAGES:            {USE_LOG_CU_WAGES}")
print(f"  USE_CONTEMP_CU:              {USE_CONTEMP_CU}")
print(f"  USE_DETRENDED_EXCESS_DEMAND: {USE_DETRENDED_EXCESS_DEMAND}")

print(f"\nInput file:  {NEW_MODEL_PATH}")
print(f"Output dir:  {OUTPUT_DIR}")
print(f"Input exists: {NEW_MODEL_PATH.exists()}")

# %% LOAD DATA
df_new = pd.read_excel(NEW_MODEL_PATH)
df_new['period'] = pd.to_datetime(df_new['period'])
df_new_full = df_new.copy()  # Keep full data for R2 values
df_new = df_new[df_new['period'] >= '2019-12-01']  # Filter to match R script

# Load BB model for comparison if available
df_bb = None
if BB_MODEL_PATH.exists():
    df_bb = pd.read_excel(BB_MODEL_PATH)
    df_bb['period'] = pd.to_datetime(df_bb['period'])
    df_bb = df_bb[df_bb['period'] >= '2019-12-01']
    print(f"\nBB Model loaded: {len(df_bb)} observations")
else:
    print("\nBB Model not found - will plot new model only")

print(f"New Model: {len(df_new)} observations from {df_new['period'].min()} to {df_new['period'].max()}")

# %% THEME SETUP - Match R script styling
def setup_plot_style():
    """Set matplotlib style to match R ggplot2 theme_bw()"""
    plt.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 17.5,
        'axes.labelsize': 16,
        'xtick.labelsize': 16,
        'ytick.labelsize': 16,
        'legend.fontsize': 12,
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'axes.edgecolor': 'black',
        'axes.linewidth': 1,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linewidth': 0.5,
    })

setup_plot_style()

# %% PLOT FUNCTION
def plot_pred_vs_actual(df, actual_col, pred_col, title, filename,
                        ylim=None, y_breaks=None, aspect_ratio=0.15,
                        end_date=None, df_bb=None, bb_pred_col=None, y_label='Percent'):
    """
    Plot actual vs predicted values, matching R script style.

    Parameters:
    -----------
    df : DataFrame - New model data
    actual_col : str - Column name for actual values
    pred_col : str - Column name for predicted values
    title : str - Plot title
    filename : str - Output filename (without extension)
    ylim : tuple - Y-axis limits (min, max)
    y_breaks : list - Y-axis tick locations
    aspect_ratio : float - Aspect ratio for coord_fixed equivalent
    end_date : str - Optional end date filter (e.g., '2023-01-01')
    df_bb : DataFrame - BB model data for comparison (optional)
    bb_pred_col : str - BB model prediction column (optional, defaults to pred_col)
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Filter by end date if specified
    plot_df = df[df['period'] <= end_date] if end_date else df

    # Plot actual
    ax.plot(plot_df['period'], plot_df[actual_col],
            color='darkblue', linewidth=1.25, label='Actual')

    # Plot new model prediction
    ax.plot(plot_df['period'], plot_df[pred_col],
            color='darkgreen', linewidth=1.25, label='New Model')

    # Plot BB model prediction if available
    if df_bb is not None:
        bb_col = bb_pred_col if bb_pred_col else pred_col
        if bb_col in df_bb.columns:
            bb_plot = df_bb[df_bb['period'] <= end_date] if end_date else df_bb
            ax.plot(bb_plot['period'], bb_plot[bb_col],
                    color='darkred', linewidth=1.25, linestyle='--', label='BB Model')

    # Title and labels
    ax.set_title(title, fontsize=17.5, fontweight='normal')
    ax.set_ylabel(y_label, fontsize=16)

    # Y-axis limits and breaks
    if ylim:
        ax.set_ylim(ylim)
    if y_breaks:
        ax.set_yticks(y_breaks)

    # X-axis formatting - quarterly dates (e.g., "2021 Q1")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(quarter_formatter))
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7]))
    plt.xticks(rotation=0)

    # Grid styling - horizontal only (matching R: panel.grid.major.x = element_blank())
    ax.yaxis.grid(True, linewidth=0.5, color='lightgray')
    ax.xaxis.grid(False)

    # Remove top and right spines (matching R: panel.border = element_blank())
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Legend at bottom (matching R: legend.position = "bottom")
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12),
              ncol=3, frameon=True, edgecolor='black', fancybox=False)

    plt.tight_layout()

    # Save figures
    plt.savefig(OUTPUT_DIR / f'{filename}.png', dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / f'{filename}.pdf', bbox_inches='tight')
    plt.show()
    print(f"  Saved: {filename}")


# %% GENERATE PLOTS
print("\nGenerating plots...")

# Figure 3: Wage Growth
plot_pred_vs_actual(
    df_new, 'gw', 'gwf1',
    'Wage growth',
    'figure_3_wage_growth_new',
    aspect_ratio=0.15,
    df_bb=df_bb
)

# Figure 7: Inflation
plot_pred_vs_actual(
    df_new, 'gcpi', 'gcpif',
    'Inflation',
    'figure_7_inflation_new',
    ylim=(-2, 12),
    y_breaks=[-2, 0, 2, 4, 6, 8, 10, 12],
    aspect_ratio=0.08,
    df_bb=df_bb
)

# Figure 8: Short-run Inflation Expectations
plot_pred_vs_actual(
    df_new, 'cf1', 'cf1f',
    'Short-run inflation expectations',
    'figure_8_short_run_exp_new',
    aspect_ratio=0.35,
    end_date='2023-01-01',
    df_bb=df_bb
)

# Figure 9: Long-run Inflation Expectations
plot_pred_vs_actual(
    df_new, 'cf10', 'cf10f',
    'Long-run inflation expectations',
    'figure_9_long_run_exp_new',
    ylim=(1, 2.5),
    y_breaks=[1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4],
    aspect_ratio=1.0,
    end_date='2023-01-01',
    df_bb=df_bb
)

# New Model specific: Shortage prediction (endogenous)
if 'shortagef' in df_new.columns:
    plot_pred_vs_actual(
        df_new, 'shortage', 'shortagef',
        'Shortage index',
        'figure_shortage_new',
        aspect_ratio=0.15,
        y_label='Index',
    )

# %% RMSE COMPARISON
def rmse(actual, pred):
    """Calculate Root Mean Square Error"""
    valid = actual.notna() & pred.notna()
    if valid.sum() == 0:
        return np.nan
    return np.sqrt(np.mean((actual[valid] - pred[valid])**2))

print("\n" + "="*70)
print(f"RMSE COMPARISON: New Model ({SPEC_SHORT_NAME}) vs BB Model")
print("="*70)
print(f"{'Variable':<30} {'New Model':<12} {'BB Model':<12} {'Improvement':<12}")
print("-"*70)

comparisons = [
    ('gw', 'gwf1', 'Wage Growth'),
    ('gcpi', 'gcpif', 'Inflation'),
    ('cf1', 'cf1f', 'Short-run Expectations'),
    ('cf10', 'cf10f', 'Long-run Expectations'),
]

for col, pred, name in comparisons:
    if col in df_new.columns and pred in df_new.columns:
        new_rmse = rmse(df_new[col], df_new[pred])

        if df_bb is not None and col in df_bb.columns and pred in df_bb.columns:
            bb_rmse = rmse(df_bb[col], df_bb[pred])
            if bb_rmse and bb_rmse > 0:
                improvement = f"{(bb_rmse - new_rmse)/bb_rmse*100:+.1f}%"
            else:
                improvement = "N/A"
            bb_str = f"{bb_rmse:.4f}"
        else:
            bb_str = "N/A"
            improvement = "N/A"

        print(f"{name:<30} {new_rmse:<12.4f} {bb_str:<12} {improvement:<12}")

# New model specific RMSE
if 'shortagef' in df_new.columns:
    shortage_rmse = rmse(df_new['shortage'], df_new['shortagef'])
    print(f"{'Shortage (New Model only)':<30} {shortage_rmse:<12.4f} {'N/A':<12} {'N/A':<12}")

print("-"*70)

# %% R-SQUARED VALUES
print("\n" + "="*70)
print("R-SQUARED VALUES (from regression output)")
print("="*70)

r2_cols = ['r2_wage', 'r2_shortage', 'r2_price', 'r2_cf1', 'r2_cf10']
r2_names = ['Wage Equation', 'Shortage Equation', 'Price Equation',
            'Short-run Exp Equation', 'Long-run Exp Equation']

for col, name in zip(r2_cols, r2_names):
    if col in df_new_full.columns:
        r2_val = df_new_full[col].iloc[0]
        print(f"{name:<30} R2 = {r2_val:.4f}")

# %% SUMMARY
print("\n" + "="*70)
print("PLOTTING COMPLETE!")
print("="*70)
print(f"\nOutput files saved to: {OUTPUT_DIR}")
print("  - figure_3_wage_growth_new.png/pdf")
print("  - figure_7_inflation_new.png/pdf")
print("  - figure_8_short_run_exp_new.png/pdf")
print("  - figure_9_long_run_exp_new.png/pdf")
print("  - figure_shortage_new.png/pdf")
print("\n")
# %%
