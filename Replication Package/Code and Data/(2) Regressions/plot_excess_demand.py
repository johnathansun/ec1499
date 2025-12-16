#!/usr/bin/env python3
"""
Plot excess demand variable from the new model.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Input
input_path = Path("/Users/johnathansun/Documents/ec1499/Replication Package/Code and Data/(1) Data/Public Data/Regression_Data.xlsx")

# Load data
df = pd.read_excel(input_path)

# Convert Date
if 'Date' in df.columns:
    df['period'] = pd.to_datetime(df['Date'])
    df = df.drop('Date', axis=1)

df = df.sort_values('period').reset_index(drop=True)

# Create variables
df['log_ngdppot'] = np.log(df['NGDPPOT'])
df['log_w'] = np.log(df['ECIWAG'])
df['log_tcu'] = np.log(df['TCU'])
df['log_tcu_trend'] = np.log(df['TCU'].rolling(window=40, min_periods=20).mean())
df['cu'] = df['log_tcu'] - df['log_tcu_trend']
df['excess_demand'] = df['log_w'] - df['log_ngdppot'] - df['cu']

# Filter to sample period
df = df[(df['period'] >= '1989-01-01') & (df['period'] <= '2023-06-30')].copy()

# Plot
fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

# Excess demand
axes[0].plot(df['period'], df['excess_demand'], 'b-', linewidth=1)
axes[0].axhline(y=0, color='k', linestyle='--', linewidth=0.5)
axes[0].set_ylabel('Excess Demand')
axes[0].set_title('Excess Demand = log(W) - log(NGDPPOT) - cu')
axes[0].grid(True, alpha=0.3)

# Detrended capacity utilization
axes[1].plot(df['period'], df['cu'], 'g-', linewidth=1)
axes[1].axhline(y=0, color='k', linestyle='--', linewidth=0.5)
axes[1].set_ylabel('cu')
axes[1].set_title('Detrended Capacity Utilization = log(TCU) - log(10yr rolling mean)')
axes[1].grid(True, alpha=0.3)

# Components
axes[2].plot(df['period'], df['log_w'] - df['log_ngdppot'], 'r-', linewidth=1, label='log(W/NGDPPOT)')
axes[2].set_ylabel('log(W/NGDPPOT)')
axes[2].set_title('Wage relative to Nominal Potential GDP')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(Path(input_path).parent.parent / '(2) Regressions' / 'excess_demand_plot.png', dpi=150)
plt.show()

print(f"\nExcess demand statistics:")
print(f"  Mean: {df['excess_demand'].mean():.4f}")
print(f"  Std:  {df['excess_demand'].std():.4f}")
print(f"  Min:  {df['excess_demand'].min():.4f}")
print(f"  Max:  {df['excess_demand'].max():.4f}")
