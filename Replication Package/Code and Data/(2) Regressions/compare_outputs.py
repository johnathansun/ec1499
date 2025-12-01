#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare Python replication outputs with Stata outputs
Handles potential differences in row/column ordering
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Paths
stata_dir = Path("/Users/johnathansun/Documents/ec1499/Replication Package/Code and Data/(2) Regressions/Output Data (Full Sample)")
python_dir = Path("/Users/johnathansun/Documents/ec1499/Replication Package/Code and Data/(2) Regressions/Output Data Python (Full Sample)")

# Tolerance for numerical comparisons
RTOL = 1e-5  # Relative tolerance
ATOL = 1e-7  # Absolute tolerance

def compare_coefficients(stata_file, python_file, sheet_name):
    """
    Compare coefficient files allowing for row reordering

    Parameters:
    -----------
    stata_file : Path
        Path to Stata output file
    python_file : Path
        Path to Python output file
    sheet_name : str
        Name of sheet to compare

    Returns:
    --------
    dict : Comparison results
    """
    print(f"\n{'='*80}")
    print(f"Comparing coefficients for {sheet_name}")
    print(f"{'='*80}")

    # Read both files
    try:
        stata_df = pd.read_excel(stata_file, sheet_name=sheet_name)
        python_df = pd.read_excel(python_file, sheet_name=sheet_name)
    except Exception as e:
        print(f"ERROR: Could not read files for sheet '{sheet_name}': {e}")
        return {'status': 'error', 'message': str(e)}

    # Get variable column name (might be different)
    var_col_stata = stata_df.columns[0]
    var_col_python = python_df.columns[0]

    print(f"Stata file has {len(stata_df)} rows")
    print(f"Python file has {len(python_df)} rows")

    # Sort both by variable name for comparison
    stata_df = stata_df.sort_values(by=var_col_stata).reset_index(drop=True)
    python_df = python_df.sort_values(by=var_col_python).reset_index(drop=True)

    # Check if variable names match
    stata_vars = set(stata_df[var_col_stata].astype(str))
    python_vars = set(python_df[var_col_python].astype(str))

    only_in_stata = stata_vars - python_vars
    only_in_python = python_vars - stata_vars

    if only_in_stata:
        print(f"WARNING: Variables only in Stata: {only_in_stata}")
    if only_in_python:
        print(f"WARNING: Variables only in Python: {only_in_python}")

    # Merge on variable name to compare coefficients
    merged = pd.merge(
        stata_df,
        python_df,
        left_on=var_col_stata,
        right_on=var_col_python,
        how='outer',
        suffixes=('_stata', '_python')
    )

    # Find coefficient columns (typically 'beta' or similar)
    coef_cols_stata = [col for col in stata_df.columns if col != var_col_stata]
    coef_cols_python = [col for col in python_df.columns if col != var_col_python]

    results = {
        'sheet': sheet_name,
        'n_vars': len(stata_vars & python_vars),
        'differences': []
    }

    # Compare coefficient values
    for stata_col in coef_cols_stata:
        # Find corresponding Python column
        python_col = stata_col if stata_col in python_df.columns else coef_cols_python[0]

        stata_col_merged = f"{stata_col}_stata" if f"{stata_col}_stata" in merged.columns else stata_col
        python_col_merged = f"{python_col}_python" if f"{python_col}_python" in merged.columns else python_col

        if stata_col_merged not in merged.columns or python_col_merged not in merged.columns:
            continue

        # Convert to numeric
        merged[stata_col_merged] = pd.to_numeric(merged[stata_col_merged], errors='coerce')
        merged[python_col_merged] = pd.to_numeric(merged[python_col_merged], errors='coerce')

        # Calculate differences
        diff = merged[stata_col_merged] - merged[python_col_merged]

        # Find significant differences
        significant_diff = ~np.isclose(
            merged[stata_col_merged],
            merged[python_col_merged],
            rtol=RTOL,
            atol=ATOL,
            equal_nan=True
        )

        n_different = significant_diff.sum()

        if n_different > 0:
            print(f"\n⚠️  WARNING: {n_different} differences found in column '{stata_col}'")

            # Show the differences
            diff_rows = merged[significant_diff][[var_col_stata, stata_col_merged, python_col_merged]].copy()
            diff_rows['difference'] = diff[significant_diff]
            diff_rows['rel_diff_%'] = 100 * diff[significant_diff] / merged[stata_col_merged][significant_diff]

            print(diff_rows.to_string(index=False))

            results['differences'].append({
                'column': stata_col,
                'n_different': int(n_different),
                'max_abs_diff': float(diff.abs().max()),
                'mean_abs_diff': float(diff.abs().mean())
            })
        else:
            print(f"✓ Column '{stata_col}': All values match within tolerance")

    if not results['differences']:
        print(f"\n✓✓✓ ALL COEFFICIENTS MATCH for {sheet_name} ✓✓✓")
        results['status'] = 'match'
    else:
        print(f"\n⚠️⚠️⚠️ DIFFERENCES FOUND in {sheet_name} ⚠️⚠️⚠️")
        results['status'] = 'differences'

    return results


def compare_summary_stats(stata_file, python_file, sheet_name):
    """
    Compare summary statistics files

    Parameters:
    -----------
    stata_file : Path
        Path to Stata output file
    python_file : Path
        Path to Python output file
    sheet_name : str
        Name of sheet to compare

    Returns:
    --------
    dict : Comparison results
    """
    print(f"\n{'='*80}")
    print(f"Comparing summary statistics for {sheet_name}")
    print(f"{'='*80}")

    try:
        stata_df = pd.read_excel(stata_file, sheet_name=sheet_name)
        python_df = pd.read_excel(python_file, sheet_name=sheet_name)
    except Exception as e:
        print(f"ERROR: Could not read files for sheet '{sheet_name}': {e}")
        return {'status': 'error', 'message': str(e)}

    print(f"Stata file has {len(stata_df)} rows, {len(stata_df.columns)} columns")
    print(f"Python file has {len(python_df)} rows, {len(python_df.columns)} columns")

    results = {
        'sheet': sheet_name,
        'differences': []
    }

    # Compare each numeric column
    var_col = stata_df.columns[0]

    for col in stata_df.columns[1:]:
        if col not in python_df.columns:
            print(f"WARNING: Column '{col}' not found in Python output")
            continue

        # Convert to numeric
        stata_vals = pd.to_numeric(stata_df[col], errors='coerce')
        python_vals = pd.to_numeric(python_df[col], errors='coerce')

        # Compare
        diff = stata_vals - python_vals

        significant_diff = ~np.isclose(
            stata_vals,
            python_vals,
            rtol=RTOL,
            atol=ATOL,
            equal_nan=True
        )

        n_different = significant_diff.sum()

        if n_different > 0:
            print(f"\n⚠️  WARNING: {n_different} differences in column '{col}'")

            diff_rows = pd.DataFrame({
                'Variable': stata_df[var_col][significant_diff],
                'Stata': stata_vals[significant_diff],
                'Python': python_vals[significant_diff],
                'Difference': diff[significant_diff]
            })

            print(diff_rows.to_string(index=False))

            results['differences'].append({
                'column': col,
                'n_different': int(n_different),
                'max_abs_diff': float(diff.abs().max())
            })
        else:
            print(f"✓ Column '{col}': All values match")

    if not results['differences']:
        print(f"\n✓✓✓ ALL SUMMARY STATISTICS MATCH for {sheet_name} ✓✓✓")
        results['status'] = 'match'
    else:
        print(f"\n⚠️⚠️⚠️ DIFFERENCES FOUND in {sheet_name} ⚠️⚠️⚠️")
        results['status'] = 'differences'

    return results


def compare_simulation_data(stata_file, python_file):
    """
    Compare simulation data files allowing for column reordering

    Parameters:
    -----------
    stata_file : Path
        Path to Stata output file
    python_file : Path
        Path to Python output file

    Returns:
    --------
    dict : Comparison results
    """
    print(f"\n{'='*80}")
    print(f"Comparing simulation data")
    print(f"{'='*80}")

    try:
        stata_df = pd.read_excel(stata_file)
        python_df = pd.read_excel(python_file)
    except Exception as e:
        print(f"ERROR: Could not read files: {e}")
        return {'status': 'error', 'message': str(e)}

    print(f"Stata file: {len(stata_df)} rows, {len(stata_df.columns)} columns")
    print(f"Python file: {len(python_df)} rows, {len(python_df.columns)} columns")

    # Check for common columns
    stata_cols = set(stata_df.columns)
    python_cols = set(python_df.columns)

    only_in_stata = stata_cols - python_cols
    only_in_python = python_cols - stata_cols
    common_cols = stata_cols & python_cols

    if only_in_stata:
        print(f"\nColumns only in Stata output: {only_in_stata}")
    if only_in_python:
        print(f"\nColumns only in Python output: {only_in_python}")

    print(f"\n{len(common_cols)} common columns to compare")

    results = {
        'differences': [],
        'n_common_cols': len(common_cols)
    }

    # Compare common columns
    for col in sorted(common_cols):
        # Skip if not numeric
        if not (pd.api.types.is_numeric_dtype(stata_df[col]) or
                pd.api.types.is_numeric_dtype(python_df[col])):
            # For non-numeric, check exact match
            if col == 'period':
                # Convert to datetime for comparison
                stata_dates = pd.to_datetime(stata_df[col])
                python_dates = pd.to_datetime(python_df[col])

                if not stata_dates.equals(python_dates):
                    print(f"⚠️  WARNING: Period column differs")
                    results['differences'].append({'column': col, 'type': 'non-numeric'})
                else:
                    print(f"✓ Column '{col}': Matches")
            continue

        # Convert to numeric
        stata_vals = pd.to_numeric(stata_df[col], errors='coerce')
        python_vals = pd.to_numeric(python_df[col], errors='coerce')

        # Check lengths match
        if len(stata_vals) != len(python_vals):
            print(f"⚠️  WARNING: Column '{col}' has different lengths ({len(stata_vals)} vs {len(python_vals)})")
            results['differences'].append({
                'column': col,
                'issue': 'length_mismatch',
                'stata_len': len(stata_vals),
                'python_len': len(python_vals)
            })
            continue

        # Compare values
        diff = stata_vals - python_vals

        # Handle NaN values
        both_nan = stata_vals.isna() & python_vals.isna()

        significant_diff = ~np.isclose(
            stata_vals,
            python_vals,
            rtol=RTOL,
            atol=ATOL,
            equal_nan=True
        )

        n_different = significant_diff.sum()

        if n_different > 0:
            print(f"\n⚠️  Column '{col}': {n_different} differences found")
            print(f"   Max absolute difference: {diff.abs().max():.10f}")
            print(f"   Mean absolute difference: {diff.abs().mean():.10f}")

            # Show first few differences
            diff_indices = np.where(significant_diff)[0][:5]
            if len(diff_indices) > 0:
                print(f"\n   First few differences:")
                for idx in diff_indices:
                    print(f"   Row {idx}: Stata={stata_vals.iloc[idx]:.10f}, "
                          f"Python={python_vals.iloc[idx]:.10f}, "
                          f"Diff={diff.iloc[idx]:.10f}")

            results['differences'].append({
                'column': col,
                'n_different': int(n_different),
                'max_abs_diff': float(diff.abs().max()),
                'mean_abs_diff': float(diff.abs().mean())
            })
        else:
            print(f"✓ Column '{col}': All {len(stata_vals)} values match (or both NaN)")

    if not results['differences']:
        print(f"\n✓✓✓ ALL SIMULATION DATA MATCHES ✓✓✓")
        results['status'] = 'match'
    else:
        print(f"\n⚠️⚠️⚠️ DIFFERENCES FOUND IN SIMULATION DATA ⚠️⚠️⚠️")
        results['status'] = 'differences'

    return results


def main():
    """Main comparison function"""

    print("="*80)
    print("COMPARING STATA AND PYTHON REGRESSION OUTPUTS")
    print("="*80)
    print(f"\nStata directory: {stata_dir}")
    print(f"Python directory: {python_dir}")
    print(f"\nNumerical tolerance: rtol={RTOL}, atol={ATOL}")

    all_results = []

    # 1. Compare coefficients
    print("\n\n" + "="*80)
    print("PART 1: COMPARING COEFFICIENTS")
    print("="*80)

    stata_coef_file = stata_dir / "eq_coefficients.xlsx"
    python_coef_file = python_dir / "eq_coefficients_python.xlsx"

    if stata_coef_file.exists() and python_coef_file.exists():
        for sheet in ['gw', 'gcpi', 'cf1', 'cf10']:
            result = compare_coefficients(stata_coef_file, python_coef_file, sheet)
            all_results.append(result)
    else:
        print(f"ERROR: Coefficient files not found")
        if not stata_coef_file.exists():
            print(f"  Missing: {stata_coef_file}")
        if not python_coef_file.exists():
            print(f"  Missing: {python_coef_file}")

    # 2. Compare summary statistics
    print("\n\n" + "="*80)
    print("PART 2: COMPARING SUMMARY STATISTICS")
    print("="*80)

    stata_summary_file = stata_dir / "summary_stats_full.xlsx"
    python_summary_file = python_dir / "summary_stats_full_python.xlsx"

    if stata_summary_file.exists() and python_summary_file.exists():
        for sheet in ['gw', 'gcpi', 'cf1', 'cf10']:
            result = compare_summary_stats(stata_summary_file, python_summary_file, sheet)
            all_results.append(result)
    else:
        print(f"ERROR: Summary statistics files not found")
        if not stata_summary_file.exists():
            print(f"  Missing: {stata_summary_file}")
        if not python_summary_file.exists():
            print(f"  Missing: {python_summary_file}")

    # 3. Compare simulation data
    print("\n\n" + "="*80)
    print("PART 3: COMPARING SIMULATION DATA")
    print("="*80)

    # Look for both .xlsx and .xls extensions
    stata_sim_file = None
    for ext in ['.xlsx', '.xls']:
        candidate = stata_dir / f"eq_simulations_data{ext}"
        if candidate.exists():
            stata_sim_file = candidate
            break

    python_sim_file = python_dir / "eq_simulations_data_python.xlsx"

    if stata_sim_file and stata_sim_file.exists() and python_sim_file.exists():
        result = compare_simulation_data(stata_sim_file, python_sim_file)
        all_results.append(result)
    else:
        print(f"ERROR: Simulation data files not found")
        if not stata_sim_file or not stata_sim_file.exists():
            print(f"  Missing: Stata simulation data file")
        if not python_sim_file.exists():
            print(f"  Missing: {python_sim_file}")

    # Final summary
    print("\n\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)

    n_match = sum(1 for r in all_results if r.get('status') == 'match')
    n_diff = sum(1 for r in all_results if r.get('status') == 'differences')
    n_error = sum(1 for r in all_results if r.get('status') == 'error')

    print(f"\nTotal comparisons: {len(all_results)}")
    print(f"  ✓ Matches: {n_match}")
    print(f"  ⚠️  Differences found: {n_diff}")
    print(f"  ❌ Errors: {n_error}")

    if n_diff > 0:
        print("\n⚠️  COMPARISONS WITH DIFFERENCES:")
        for r in all_results:
            if r.get('status') == 'differences':
                sheet = r.get('sheet', 'simulation data')
                print(f"\n  - {sheet}:")
                for diff in r.get('differences', []):
                    if 'column' in diff:
                        print(f"    • Column '{diff['column']}': {diff.get('n_different', 'N/A')} differences")
                        if 'max_abs_diff' in diff:
                            print(f"      Max difference: {diff['max_abs_diff']:.2e}")

    if n_match == len(all_results):
        print("\n" + "="*80)
        print("🎉 SUCCESS! ALL OUTPUTS MATCH WITHIN TOLERANCE! 🎉")
        print("="*80)
    elif n_diff > 0:
        print("\n" + "="*80)
        print("⚠️  REVIEW NEEDED: Some differences found")
        print("="*80)

    return all_results


if __name__ == "__main__":
    results = main()
