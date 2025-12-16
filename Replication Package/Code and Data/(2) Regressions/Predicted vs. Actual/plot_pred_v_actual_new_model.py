# %%
"""
Predicted vs. Actual Plots: New Model vs. BB Model
Liang & Sun (2025)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

# %% CONFIG
NEW_MODEL_PATH = Path("/Users/johnathansun/Documents/ec1499/Replication Package/Code and Data/(2) Regressions/Output Data Python (New Model Pre Covid)/eq_simulations_data_new_model_pre_covid.xlsx")
BB_MODEL_PATH = Path("/Users/johnathansun/Documents/ec1499/Replication Package/Code and Data/(2) Regressions/Output Data (Pre Covid Sample)/eq_simulations_data_restricted.xls")
OUTPUT_DIR = Path("/Users/johnathansun/Documents/ec1499/Replication Package/Code and Data/(2) Regressions/Predicted vs. Actual/Figures Python (New Model)")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# %% LOAD DATA
df_new = pd.read_excel(NEW_MODEL_PATH)
df_new['period'] = pd.to_datetime(df_new['period'])
df_new = df_new[df_new['period'] >= '2020-01-01']

df_bb = pd.read_excel(BB_MODEL_PATH) if BB_MODEL_PATH.exists() else None
if df_bb is not None:
    df_bb['period'] = pd.to_datetime(df_bb['period'])
    df_bb = df_bb[df_bb['period'] >= '2020-01-01']

# %% PLOT FUNCTION
def plot_comparison(actual_col, pred_col, title, filename, ylim=None, end_date=None):
    """Plot actual vs predicted for both models."""
    fig, ax = plt.subplots(figsize=(10, 6))

    df = df_new[df_new['period'] <= end_date] if end_date else df_new
    bb = df_bb[df_bb['period'] <= end_date] if (df_bb is not None and end_date) else df_bb

    ax.plot(df['period'], df[actual_col], 'darkblue', lw=2, label='Actual')
    ax.plot(df['period'], df[pred_col], 'darkgreen', lw=1.5, label='New Model')
    if bb is not None and pred_col in bb.columns:
        ax.plot(bb['period'], bb[pred_col], 'darkred', lw=1.5, ls='--', label='BB Model')

    ax.set_title(title)
    ax.set_ylabel('Percent')
    if ylim: ax.set_ylim(ylim)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y Q%q'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7]))
    ax.spines[['top', 'right']].set_visible(False)
    ax.yaxis.grid(True, lw=0.5, color='lightgray')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3, frameon=True)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'{filename}.png', dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / f'{filename}.pdf', bbox_inches='tight')
    plt.show()
    print(f"Saved: {filename}")

# %% GENERATE PLOTS
plot_comparison('gw', 'gwf1', 'Figure 3. WAGE GROWTH', 'figure_3_wage_growth')
plot_comparison('shortage', 'shortagef', 'SHORTAGE (New Model)', 'figure_shortage')
plot_comparison('gcpi', 'gcpif', 'Figure 7. INFLATION', 'figure_7_inflation', ylim=(-2, 12))
plot_comparison('cf1', 'cf1f', 'Figure 8. SHORT-RUN EXPECTATIONS', 'figure_8_short_run_exp', end_date='2023-01-01')
plot_comparison('cf10', 'cf10f', 'Figure 9. LONG-RUN EXPECTATIONS', 'figure_9_long_run_exp', ylim=(1, 2.5), end_date='2023-01-01')

# %% RMSE COMPARISON
def rmse(actual, pred):
    valid = actual.notna() & pred.notna()
    return np.sqrt(np.mean((actual[valid] - pred[valid])**2))

print("\n" + "="*60)
print("RMSE COMPARISON: New Model vs BB Model")
print("="*60)
print(f"{'Variable':<25} {'New Model':<12} {'BB Model':<12} {'Diff':<10}")
print("-"*60)

for col, pred, name in [('gw','gwf1','Wage Growth'), ('gcpi','gcpif','Inflation'),
                        ('cf1','cf1f','Short-run Exp'), ('cf10','cf10f','Long-run Exp')]:
    new_rmse = rmse(df_new[col], df_new[pred])
    bb_rmse = rmse(df_bb[col], df_bb[pred]) if df_bb is not None else None
    diff = f"{(bb_rmse - new_rmse)/bb_rmse*100:+.1f}%" if bb_rmse else "N/A"
    bb_str = f"{bb_rmse:.4f}" if bb_rmse else "N/A"
    print(f"{name:<25} {new_rmse:<12.4f} {bb_str:<12} {diff:<10}")

print("\nDone!")
# %%
