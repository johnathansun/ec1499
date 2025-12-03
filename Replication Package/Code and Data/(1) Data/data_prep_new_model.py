# %%
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data Preparation for Modified Bernanke-Blanchard Model
Liang & Sun (2025)

This script downloads new variables required for the modified model:
1. TCU - Total Capacity Utilization (FRED)
2. NGDPPOT - Nominal Potential GDP (FRED)
3. GSCPI - Global Supply Chain Pressure Index (NY Fed)

And merges them with the existing Regression_Data.xlsx

Required packages:
    pip install fredapi requests openpyxl
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import fredapi
try:
    from fredapi import Fred
    HAS_FREDAPI = True
except ImportError:
    HAS_FREDAPI = False
    print("ERROR: fredapi not installed. Install with: pip install fredapi")

# Import requests for NY Fed data
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False
    print("ERROR: requests not installed. Install with: pip install requests")

# %%
#****************************CHANGE PATHS HERE************************************

# Input: Original regression data
input_path = Path("/Users/johnathansun/Documents/ec1499/Replication Package/Code and Data/(1) Data/Public Data/Regression_Data.xlsx")

# Output: New regression data with additional variables
output_dir = Path("/Users/johnathansun/Documents/ec1499/Replication Package/Code and Data/(1) Data/Public Data")
output_file = output_dir / "Regression_Data_NewModel.xlsx"

# FRED API key - GET YOUR FREE KEY AT: https://fred.stlouisfed.org/docs/api/api_key.html
# You can also set environment variable FRED_API_KEY instead
FRED_API_KEY = "your_api_key_here"  # <-- REPLACE WITH YOUR API KEY

#*********************************************************************************

# %%
# =============================================================================
# INITIALIZE FRED API
# =============================================================================

fred = None
if HAS_FREDAPI:
    try:
        # Try to initialize Fred - will use FRED_API_KEY env var if api_key not provided
        if FRED_API_KEY and FRED_API_KEY != "your_api_key_here":
            fred = Fred(api_key=FRED_API_KEY)
        else:
            # Try environment variable
            import os
            if 'FRED_API_KEY' in os.environ:
                fred = Fred()
            else:
                print("WARNING: No FRED API key provided.")
                print("         Get a free key at: https://fred.stlouisfed.org/docs/api/api_key.html")
                print("         Then set FRED_API_KEY variable in this script or as environment variable.")
    except Exception as e:
        print(f"ERROR initializing FRED API: {e}")


# %%
# =============================================================================
# DOWNLOAD FRED DATA
# =============================================================================

def download_fred_series(series_id, start_date='1980-01-01'):
    """
    Download a series from FRED using fredapi

    Parameters:
    -----------
    series_id : str
        FRED series ID (e.g., 'TCU', 'NGDPPOT')
    start_date : str
        Start date in 'YYYY-MM-DD' format

    Returns:
    --------
    pd.Series with date index
    """
    if fred is None:
        print(f"  Cannot download {series_id}: FRED API not initialized")
        return None

    try:
        # Get the series
        data = fred.get_series(series_id, observation_start=start_date)
        print(f"  Downloaded {series_id}: {len(data)} observations")
        print(f"    Date range: {data.index.min().strftime('%Y-%m-%d')} to {data.index.max().strftime('%Y-%m-%d')}")
        return data
    except Exception as e:
        print(f"  Error downloading {series_id}: {e}")
        return None


def search_fred(search_text, limit=10):
    """
    Search for series on FRED

    Parameters:
    -----------
    search_text : str
        Text to search for
    limit : int
        Maximum number of results

    Returns:
    --------
    pd.DataFrame with search results
    """
    if fred is None:
        print("Cannot search: FRED API not initialized")
        return None

    try:
        results = fred.search(search_text, limit=limit)
        return results
    except Exception as e:
        print(f"Error searching FRED: {e}")
        return None


# %%
# =============================================================================
# DOWNLOAD NY FED GSCPI DATA
# =============================================================================

def download_gscpi():
    """
    Download Global Supply Chain Pressure Index from NY Fed

    The GSCPI is available at:
    https://www.newyorkfed.org/research/policy/gscpi

    Returns:
    --------
    pd.Series with date index and GSCPI values
    """
    if not HAS_REQUESTS:
        print("  Cannot download GSCPI: requests not available")
        return None

    # NY Fed provides the data as an Excel file
    url = "https://www.newyorkfed.org/medialibrary/research/interactives/gscpi/downloads/gscpi_data.xlsx"

    try:
        print("  Downloading GSCPI from NY Fed...")
        response = requests.get(url, timeout=30)

        if response.status_code == 200:
            # Read Excel from bytes
            from io import BytesIO
            df = pd.read_excel(BytesIO(response.content))

            # Clean column names
            df.columns = df.columns.str.strip()
            print(f"    Raw columns: {df.columns.tolist()}")

            # Find the date column (usually first column or named 'Date')
            date_col = None
            for col in df.columns:
                if 'date' in col.lower() or col == df.columns[0]:
                    date_col = col
                    break

            # Find the GSCPI column (the main index, not individual factors)
            gscpi_col = None
            for col in df.columns:
                col_lower = col.lower()
                if 'gscpi' in col_lower and 'factor' not in col_lower:
                    gscpi_col = col
                    break

            # If not found, try looking for a column that's just numeric
            if gscpi_col is None:
                for col in df.columns:
                    if col != date_col and pd.api.types.is_numeric_dtype(df[col]):
                        gscpi_col = col
                        break

            if date_col and gscpi_col:
                df['date'] = pd.to_datetime(df[date_col])
                df['GSCPI'] = pd.to_numeric(df[gscpi_col], errors='coerce')
                result = df[['date', 'GSCPI']].dropna().set_index('date')['GSCPI']
                print(f"  Downloaded GSCPI: {len(result)} observations")
                print(f"    Date range: {result.index.min().strftime('%Y-%m-%d')} to {result.index.max().strftime('%Y-%m-%d')}")
                return result
            else:
                print(f"  Could not identify columns. Date col: {date_col}, GSCPI col: {gscpi_col}")
                print(f"  Available columns: {df.columns.tolist()}")
                return None
        else:
            print(f"  Failed to download GSCPI: HTTP {response.status_code}")
            return None

    except Exception as e:
        print(f"  Error downloading GSCPI: {e}")
        return None


# %%
# =============================================================================
# CONVERT TO QUARTERLY
# =============================================================================

def to_quarterly(series, method='mean'):
    """
    Convert a time series to quarterly frequency

    Parameters:
    -----------
    series : pd.Series
        Series with datetime index
    method : str
        'last' - take last observation of quarter
        'mean' - take average of quarter
        'first' - take first observation of quarter

    Returns:
    --------
    pd.Series with quarterly frequency (end-of-quarter dates)
    """
    if method == 'last':
        return series.resample('QE').last()
    elif method == 'mean':
        return series.resample('QE').mean()
    elif method == 'first':
        return series.resample('QE').first()
    else:
        raise ValueError(f"Unknown method: {method}")


# %%
# =============================================================================
# MAIN DATA PREPARATION
# =============================================================================

print("="*80)
print("DATA PREPARATION FOR NEW MODEL")
print("="*80)

# Load existing regression data
print("\n1. Loading existing regression data...")
try:
    df_original = pd.read_excel(input_path)
    print(f"   Loaded {len(df_original)} observations")
    print(f"   Columns: {df_original.columns.tolist()}")
except Exception as e:
    print(f"   Error loading data: {e}")
    df_original = None

# %%
# Download FRED data
print("\n2. Downloading FRED data...")

if fred is not None:
    # TCU - Total Capacity Utilization (Total Industry)
    print("\n   [TCU] Total Capacity Utilization:")
    tcu_series = download_fred_series('TCU', start_date='1980-01-01')

    # NGDPPOT - Nominal Potential GDP
    print("\n   [NGDPPOT] Nominal Potential GDP:")
    ngdppot_series = download_fred_series('NGDPPOT', start_date='1980-01-01')

    # If NGDPPOT fails, try alternatives
    if ngdppot_series is None:
        print("\n   Trying alternative: GDPPOT (Real Potential GDP)...")
        ngdppot_series = download_fred_series('GDPPOT', start_date='1980-01-01')
else:
    print("   Skipping FRED downloads - API not initialized")
    tcu_series = None
    ngdppot_series = None

# %%
# Download NY Fed GSCPI
print("\n3. Downloading NY Fed GSCPI...")
gscpi_series = download_gscpi()

# %%
# Convert to quarterly if needed
print("\n4. Converting to quarterly frequency...")

if tcu_series is not None:
    # TCU is monthly, convert to quarterly average
    tcu_quarterly = to_quarterly(tcu_series, method='mean')
    tcu_quarterly.name = 'TCU'
    print(f"   TCU: {len(tcu_quarterly)} quarterly observations")
else:
    tcu_quarterly = None

if ngdppot_series is not None:
    # NGDPPOT is already quarterly, but ensure proper format
    ngdppot_quarterly = to_quarterly(ngdppot_series, method='last')
    ngdppot_quarterly.name = 'NGDPPOT'
    print(f"   NGDPPOT: {len(ngdppot_quarterly)} quarterly observations")
else:
    ngdppot_quarterly = None

if gscpi_series is not None:
    # GSCPI is monthly, convert to quarterly average
    gscpi_quarterly = to_quarterly(gscpi_series, method='mean')
    gscpi_quarterly.name = 'GSCPI'
    print(f"   GSCPI: {len(gscpi_quarterly)} quarterly observations")
else:
    gscpi_quarterly = None

# %%
# Merge with original data
print("\n5. Merging with original regression data...")

if df_original is not None:
    # Convert Date column to datetime
    if 'Date' in df_original.columns:
        df_original['Date'] = pd.to_datetime(df_original['Date'].apply(
            lambda x: x.replace(" ", "") if isinstance(x, str) else x
        ))
        df_merged = df_original.set_index('Date')
    else:
        df_merged = df_original.copy()

    # Merge new variables
    if tcu_quarterly is not None:
        df_merged = df_merged.join(tcu_quarterly, how='left')
        n_valid = df_merged['TCU'].notna().sum()
        print(f"   Added TCU: {n_valid} non-null values")

    if ngdppot_quarterly is not None:
        df_merged = df_merged.join(ngdppot_quarterly, how='left')
        n_valid = df_merged['NGDPPOT'].notna().sum()
        print(f"   Added NGDPPOT: {n_valid} non-null values")

    if gscpi_quarterly is not None:
        df_merged = df_merged.join(gscpi_quarterly, how='left')
        n_valid = df_merged['GSCPI'].notna().sum()
        print(f"   Added GSCPI: {n_valid} non-null values")

    # Reset index to make Date a column again
    df_merged = df_merged.reset_index()
    df_merged = df_merged.rename(columns={'index': 'Date'})

    print(f"\n   Final dataset: {len(df_merged)} observations, {len(df_merged.columns)} columns")
else:
    df_merged = None

# %%
# Create derived variables for the new model
print("\n6. Creating derived variables...")

if df_merged is not None:
    # Log capacity utilization (TCU is in percent, e.g., 78.5 means 78.5%)
    if 'TCU' in df_merged.columns:
        df_merged['log_cu'] = np.log(df_merged['TCU'] / 100)
        print("   Created log_cu = log(TCU/100)")

    # Log nominal potential GDP
    if 'NGDPPOT' in df_merged.columns:
        df_merged['log_ngdppot'] = np.log(df_merged['NGDPPOT'])
        print("   Created log_ngdppot = log(NGDPPOT)")

    # Note about excess demand computation
    print("\n   Note: The excess demand proxy [log(w) - log(ngdppot) - log(cu)]")
    print("         will be computed in the regression script using wage levels.")

# %%
# Summary statistics for new variables
print("\n7. Summary statistics for new variables:")
print("-"*60)

if df_merged is not None:
    new_vars = ['TCU', 'NGDPPOT', 'GSCPI', 'log_cu', 'log_ngdppot']
    for var in new_vars:
        if var in df_merged.columns:
            series = df_merged[var].dropna()
            if len(series) > 0:
                print(f"\n   {var}:")
                print(f"      N:    {len(series)}")
                print(f"      Mean: {series.mean():.4f}")
                print(f"      Std:  {series.std():.4f}")
                print(f"      Min:  {series.min():.4f}")
                print(f"      Max:  {series.max():.4f}")

# %%
# Check data availability for sample period
print("\n8. Data availability check (1989 Q1 - 2023 Q2):")
print("-"*60)

if df_merged is not None:
    sample_start = '1989-01-01'
    sample_end = '2023-06-30'

    df_sample = df_merged[(df_merged['Date'] >= sample_start) & (df_merged['Date'] <= sample_end)]
    total = len(df_sample)

    new_vars = ['TCU', 'NGDPPOT', 'GSCPI']
    for var in new_vars:
        if var in df_merged.columns:
            available = df_sample[var].notna().sum()
            pct = 100 * available / total if total > 0 else 0
            first_valid = df_sample[df_sample[var].notna()]['Date'].min()
            print(f"   {var}: {available}/{total} ({pct:.1f}%) - starts {first_valid}")

    # GSCPI coverage note
    if 'GSCPI' in df_merged.columns:
        print("\n   Note: NY Fed GSCPI begins December 1997")

# %%
# Save merged dataset
print("\n9. Saving merged dataset...")

if df_merged is not None:
    # Save Excel version
    df_merged.to_excel(output_file, index=False)
    print(f"   Saved to: {output_file}")

    # Also save a CSV version
    csv_file = output_file.with_suffix('.csv')
    df_merged.to_csv(csv_file, index=False)
    print(f"   Saved to: {csv_file}")

# %%
# Print final column list
print("\n10. Final dataset columns:")
print("-"*60)
if df_merged is not None:
    for i, col in enumerate(df_merged.columns):
        new_marker = " [NEW]" if col in ['TCU', 'NGDPPOT', 'GSCPI', 'log_cu', 'log_ngdppot'] else ""
        print(f"   {i+1:2d}. {col}{new_marker}")

print("\n" + "="*80)
print("DATA PREPARATION COMPLETE!")
print("="*80)

# %%
# Optional: Create diagnostic plots
print("\n11. Creating diagnostic plots...")

try:
    import matplotlib.pyplot as plt

    # Count how many new variables we have
    plot_vars = [v for v in ['TCU', 'NGDPPOT', 'GSCPI'] if v in df_merged.columns and df_merged[v].notna().any()]

    if len(plot_vars) > 0:
        fig, axes = plt.subplots(len(plot_vars), 1, figsize=(12, 4*len(plot_vars)))
        if len(plot_vars) == 1:
            axes = [axes]

        for i, var in enumerate(plot_vars):
            ax = axes[i]
            valid_data = df_merged[['Date', var]].dropna()
            ax.plot(valid_data['Date'], valid_data[var], linewidth=1.5)

            if var == 'TCU':
                ax.set_title('Total Capacity Utilization (TCU)', fontsize=14)
                ax.set_ylabel('Percent')
                ax.axhline(y=80, color='r', linestyle='--', alpha=0.5, label='80% threshold')
                ax.legend()
            elif var == 'NGDPPOT':
                ax.set_title('Nominal Potential GDP (NGDPPOT)', fontsize=14)
                ax.set_ylabel('Billions of $')
            elif var == 'GSCPI':
                ax.set_title('Global Supply Chain Pressure Index (GSCPI)', fontsize=14)
                ax.set_ylabel('Std. Deviations from Mean')
                ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)

            ax.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save figure
        fig_path = output_dir / 'new_variables_diagnostic.png'
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        print(f"   Saved diagnostic plot to: {fig_path}")

        plt.show()
    else:
        print("   No new variables to plot")

except ImportError:
    print("   (matplotlib not available for plotting)")
except Exception as e:
    print(f"   Error creating plots: {e}")

# %%
