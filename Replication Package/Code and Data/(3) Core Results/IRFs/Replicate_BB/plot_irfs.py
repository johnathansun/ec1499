# %%
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
What Caused the U.S Pandemic-Era Inflation? Bernanke, Blanchard 2023
Python replication of plot_irfs.R
This version: December 3, 2024

This file generates plots of the IRFs from Bernanke and Blanchard (2023).

Generates:
- Figure 10: Responses of inflation to shocks to energy prices, food prices, and shortages
- Figure 11: Response of inflation to shocks to the vacancy-to-unemployment ratio
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# %%

#****************************CHANGE PATH HERE************************************
# Input Location - output from irf_simulations.py
input_dir = Path("/Users/johnathansun/Documents/ec1499/Replication Package/Code and Data/(3) Core Results/IRFs/Output Data Python")

# Output Location for figures
output_dir = Path("/Users/johnathansun/Documents/ec1499/Replication Package/Code and Data/(3) Core Results/IRFs/Figures Python")
output_dir.mkdir(parents=True, exist_ok=True)

#*********************************************************************************

print("Loading IRF results...")

# Load IRF data
irf_energy = pd.read_excel(input_dir / 'results_energy.xlsx')
irf_food = pd.read_excel(input_dir / 'results_food.xlsx')
irf_shortage = pd.read_excel(input_dir / 'results_shortage.xlsx')
irf_vu = pd.read_excel(input_dir / 'results_vu.xlsx')

print(f"Loaded {len(irf_energy)} periods of IRF data")

# %%
# Filter to periods 5-20 (quarters 1-16 after shock) to match R script
# In Python, period 5 is the first period after the shock (t=4 in 0-indexed)
irf_energy_filtered = irf_energy[(irf_energy['period'] >= 5) & (irf_energy['period'] <= 20)].copy()
irf_food_filtered = irf_food[(irf_food['period'] >= 5) & (irf_food['period'] <= 20)].copy()
irf_shortage_filtered = irf_shortage[(irf_shortage['period'] >= 5) & (irf_shortage['period'] <= 20)].copy()
irf_vu_filtered = irf_vu[(irf_vu['period'] >= 5) & (irf_vu['period'] <= 20)].copy()

# Create combined DataFrame for Figure 10
irf_data = pd.DataFrame({
    'quarter': np.arange(1, len(irf_energy_filtered) + 1),
    'energy': irf_energy_filtered['gcpi_simul'].values,
    'food': irf_food_filtered['gcpi_simul'].values,
    'shortage': irf_shortage_filtered['gcpi_simul'].values
})

print(f"Plotting {len(irf_data)} quarters of IRF responses")

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
# Figure 10: RESPONSES OF INFLATION TO SHOCKS
print("\nCreating Figure 10: Responses to Energy, Food, and Shortage Shocks...")

fig, ax = plt.subplots(figsize=(12, 7))

quarters = irf_data['quarter']

ax.plot(quarters, irf_data['energy'], color='blue', linewidth=1.5, label='Energy Prices')
ax.plot(quarters, irf_data['food'], color='red', linewidth=1.5, label='Food Prices')
ax.plot(quarters, irf_data['shortage'], color='orange', linewidth=1.5, label='Shortages')

ax.set_title('Figure 10. RESPONSES OF INFLATION TO SHOCKS TO THE RELATIVE PRICE\n'
             'OF ENERGY, THE RELATIVE PRICE OF FOOD, AND THE SHORTAGE INDEX',
             fontsize=17.5, fontweight='normal')
ax.set_xlabel('Quarter', fontsize=16)
ax.set_ylabel('Percent', fontsize=16)

# Set axis limits and ticks
ax.set_xlim(0.5, 16.5)
ax.set_xticks(range(1, 17))
ax.set_ylim(-0.5, 2.5)
ax.set_yticks(np.arange(-0.5, 3.0, 0.5))

# Gridlines - horizontal only
ax.yaxis.grid(True, linestyle='-', linewidth=0.5, color='lightgray')
ax.xaxis.grid(False)

# Remove top and right spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Legend at bottom
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=3,
          frameon=True, edgecolor='black', fancybox=False)

plt.tight_layout()
plt.savefig(output_dir / 'figure_10_irf_supply_shocks.png', dpi=300, bbox_inches='tight')
plt.savefig(output_dir / 'figure_10_irf_supply_shocks.pdf', bbox_inches='tight')
print(f"  Saved to {output_dir / 'figure_10_irf_supply_shocks.png'}")
plt.show()

# %%
# Figure 11: RESPONSE OF INFLATION TO V/U SHOCK
print("\nCreating Figure 11: Response to V/U Shock...")

fig, ax = plt.subplots(figsize=(12, 7))

quarters_vu = np.arange(1, len(irf_vu_filtered) + 1)

ax.plot(quarters_vu, irf_vu_filtered['gcpi_simul'].values, color='blue', linewidth=1.5)

ax.set_title('Figure 11. RESPONSE OF INFLATION TO SHOCKS TO THE VACANCY TO\n'
             'UNEMPLOYMENT RATIO.',
             fontsize=17.5, fontweight='normal')
ax.set_xlabel('Quarter', fontsize=16)
ax.set_ylabel('Percent', fontsize=16)

# Set axis limits and ticks
ax.set_xlim(0.5, 16.5)
ax.set_xticks(range(1, 17))
ax.set_ylim(0.0, 2.0)
ax.set_yticks(np.arange(0.0, 2.2, 0.2))

# Gridlines - horizontal only
ax.yaxis.grid(True, linestyle='-', linewidth=0.5, color='lightgray')
ax.xaxis.grid(False)

# Remove top and right spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# No legend for single line plot (matching R script)

plt.tight_layout()
plt.savefig(output_dir / 'figure_11_irf_vu_shock.png', dpi=300, bbox_inches='tight')
plt.savefig(output_dir / 'figure_11_irf_vu_shock.pdf', bbox_inches='tight')
print(f"  Saved to {output_dir / 'figure_11_irf_vu_shock.png'}")
plt.show()

# %%
# Create a combined figure with both plots
print("\nCreating combined figure...")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Figure 10
ax1 = axes[0]
ax1.plot(quarters, irf_data['energy'], color='blue', linewidth=1.5, label='Energy Prices')
ax1.plot(quarters, irf_data['food'], color='red', linewidth=1.5, label='Food Prices')
ax1.plot(quarters, irf_data['shortage'], color='orange', linewidth=1.5, label='Shortages')
ax1.set_title('Figure 10. Supply Shock Responses', fontsize=14)
ax1.set_xlabel('Quarter', fontsize=12)
ax1.set_ylabel('Percent', fontsize=12)
ax1.set_xlim(0.5, 16.5)
ax1.set_xticks(range(1, 17, 2))
ax1.set_ylim(-0.5, 2.5)
ax1.yaxis.grid(True, linestyle='-', linewidth=0.5, color='lightgray')
ax1.xaxis.grid(False)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.legend(loc='best', fontsize=10, frameon=True, edgecolor='black')

# Figure 11
ax2 = axes[1]
ax2.plot(quarters_vu, irf_vu_filtered['gcpi_simul'].values, color='blue', linewidth=1.5, label='V/U')
ax2.set_title('Figure 11. V/U Shock Response', fontsize=14)
ax2.set_xlabel('Quarter', fontsize=12)
ax2.set_ylabel('Percent', fontsize=12)
ax2.set_xlim(0.5, 16.5)
ax2.set_xticks(range(1, 17, 2))
ax2.set_ylim(0.0, 2.0)
ax2.yaxis.grid(True, linestyle='-', linewidth=0.5, color='lightgray')
ax2.xaxis.grid(False)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

plt.suptitle('Impulse Response Functions: Inflation Response to Various Shocks',
             fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(output_dir / 'figures_10_11_combined.png', dpi=300, bbox_inches='tight')
plt.savefig(output_dir / 'figures_10_11_combined.pdf', bbox_inches='tight')
print(f"  Saved to {output_dir / 'figures_10_11_combined.png'}")
plt.show()

# %%
# Print summary statistics
print("\n" + "="*80)
print("IRF SUMMARY")
print("="*80)
print(f"\nFigure 10 - Peak inflation responses (quarters 1-16 after shock):")
print(f"  Energy:   Max = {irf_data['energy'].max():.4f} at quarter {irf_data['energy'].argmax() + 1}")
print(f"  Food:     Max = {irf_data['food'].max():.4f} at quarter {irf_data['food'].argmax() + 1}")
print(f"  Shortage: Max = {irf_data['shortage'].max():.4f} at quarter {irf_data['shortage'].argmax() + 1}")

print(f"\nFigure 11 - V/U shock response:")
print(f"  V/U:      Max = {irf_vu_filtered['gcpi_simul'].max():.4f} at quarter {irf_vu_filtered['gcpi_simul'].argmax() + 1}")

print("\n" + "="*80)
print("PLOTTING COMPLETE!")
print("="*80)
print(f"\nOutput files saved to: {output_dir}")
print("  - figure_10_irf_supply_shocks.png/pdf")
print("  - figure_11_irf_vu_shock.png/pdf")
print("  - figures_10_11_combined.png/pdf")
print("\n")

# %%
