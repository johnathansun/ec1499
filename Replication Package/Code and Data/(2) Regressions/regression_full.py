# %%
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
What Caused the U.S Pandemic-Era Inflation? Bernanke, Blanchard 2023
Python replication of regression_full.do
This version: December 1, 2024

This file runs simulations (empirical model) using Regression_Data
In this file, we use CPI to calculate grpe and grpf.
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# %%

#****************************CHANGE PATH HERE************************************
# Note: change the filepath to the data folder on your computer

# Input Location
# input_path = Path("../Replication Package/Code and Data/(1) Data/Public Data/Regression_Data.xlsx")
input_path = Path("/Users/johnathansun/Documents/ec1499/Replication Package/Code and Data/(1) Data/Public Data/Regression_Data.xlsx")


# Output Location
# output_dir = Path("../Replication Package/Code and Data/(2) Regressions/Output Data (Full Sample)")
output_dir = Path("/Users/johnathansun/Documents/ec1499/Replication Package/Code and Data/(2) Regressions/Output Data Python (Full Sample)")
output_dir.mkdir(parents=True, exist_ok=True)

#*********************************************************************************

print("Loading data...")
# Load data - try both .dta and .xlsx
try:
    df = pd.read_stata(input_path.with_suffix('.dta'))
except:
    df = pd.read_excel(input_path)



# %%
# QUARTERLY TIME VARIABLE
# Convert Date to datetime if needed
if 'Date' in df.columns:
    if not pd.api.types.is_datetime64_any_dtype(df['Date']):
        df['Date'] = pd.to_datetime(df['Date'].apply(lambda x: x.replace(" ", "")))
        # df['Date'] = pd.PeriodIndex(df['Date'], freq="Q").to_timestamp()
    df['period'] = df['Date']
    df = df.drop('Date', axis=1)
else:
    # Assume period already exists
    pass

# Set period as index for easier lag operations
df = df.sort_values('period').reset_index(drop=True)

# DATA MANIPULATION
print("Creating derived variables...")

# Create growth rates (annualized quarterly: 400 * log difference)
df['gcpi'] = 400 * (np.log(df['CPIAUCSL']) - np.log(df['CPIAUCSL'].shift(1)))
df['gw'] = 400 * (np.log(df['ECIWAG']) - np.log(df['ECIWAG'].shift(1)))
df['gpty'] = 400 * (np.log(df['OPHNFB']) - np.log(df['OPHNFB'].shift(1)))

# Moving average of productivity growth (8-quarter average)
df['magpty'] = 0.125 * sum([df['gpty'].shift(i) for i in range(8)])

# Relative prices
df['rpe'] = df['CPIENGSL'] / df['ECIWAG']
df['rpf'] = df['CPIUFDSL'] / df['ECIWAG']
df['grpe'] = 400 * (np.log(df['rpe']) - np.log(df['rpe'].shift(1)))
df['grpf'] = 400 * (np.log(df['rpf']) - np.log(df['rpf'].shift(1)))

# Other variables
df['vu'] = df['VOVERU']
df['cf1'] = df['EXPINF1YR']
df['cf10'] = df['EXPINF10YR']

# Shortage (replace missing with 5)
df['shortage'] = df['SHORTAGE'].fillna(5)

# Catchup term
df['diffcpicf'] = 0.25 * sum([df['gcpi'].shift(i) for i in range(4)]) - df['cf1'].shift(4)

# Add dummy variables for COVID
df['dummyq2_2020'] = 0.0
df['dummyq3_2020'] = 0.0

# Create quarterly period identifier for comparison
df['year'] = df['period'].dt.year
df['quarter'] = df['period'].dt.quarter
df.loc[(df['year'] == 2020) & (df['quarter'] == 2), 'dummyq2_2020'] = 1.0
df.loc[(df['year'] == 2020) & (df['quarter'] == 3), 'dummyq3_2020'] = 1.0

# Keep only required variables
keep_vars = ['period', 'gcpi', 'vu', 'gw', 'magpty', 'grpe', 'grpf', 'cf1', 'cf10',
             'shortage', 'diffcpicf', 'CPIENGSL', 'CPIUFDSL', 'dummyq2_2020', 'dummyq3_2020']
df = df[keep_vars].copy()

# Filter to sample period: 1989 Q1 to 2023 Q2
df = df[(df['period'] >= '1989-01-01') & (df['period'] <= '2023-06-30')].copy()
df = df.reset_index(drop=True)

print(f"Sample size: {len(df)} observations")
print(f"Sample period: {df['period'].min()} to {df['period'].max()}")



# %%
# Helper function for constrained linear regression
def constrained_regression(y, X, constraint_matrix, constraint_value, add_constant=True):
    """
    Perform constrained OLS regression

    Parameters:
    -----------
    y : array-like
        Dependent variable
    X : DataFrame or array-like
        Independent variables
    constraint_matrix : array-like
        Matrix R in constraint R*beta = q
    constraint_value : array-like
        Vector q in constraint R*beta = q
    add_constant : bool
        Whether to add a constant term

    Returns:
    --------
    results : RegressionResults object with additional attributes
    """
    # Remove NaN values
    data = pd.DataFrame(X).copy()
    data['y'] = y
    data = data.dropna()

    y_clean = data['y'].values
    X_clean = data.drop('y', axis=1).values

    if add_constant:
        X_clean = sm.add_constant(X_clean)
        # Adjust constraint matrix to include constant term
        if constraint_matrix is not None:
            n_constraints = constraint_matrix.shape[0]
            constraint_matrix = np.column_stack([np.zeros((n_constraints, 1)), constraint_matrix])

    # Fit constrained model
    if constraint_matrix is not None and constraint_value is not None:
        model = sm.OLS(y_clean, X_clean)
        results = model.fit_regularized(L1_wt=0, alpha=0)  # Placeholder
        # For true constrained regression, use optimization
        from scipy.optimize import minimize

        def objective(beta):
            resid = y_clean - X_clean @ beta
            return np.sum(resid ** 2)

        def constraint_eq(beta):
            return constraint_matrix @ beta - constraint_value

        # Initial guess from OLS
        beta_init = np.linalg.lstsq(X_clean, y_clean, rcond=None)[0]

        # Optimize with constraint
        cons = {'type': 'eq', 'fun': constraint_eq}
        result = minimize(objective, beta_init, method='SLSQP', constraints=cons)

        # Create results object
        beta = result.x
        y_pred = X_clean @ beta
        residuals = y_clean - y_pred

        # Calculate standard errors and other statistics
        n = len(y_clean)
        k = len(beta)
        mse = np.sum(residuals ** 2) / (n - k)

        # Covariance matrix for constrained estimator
        # Using formula: (X'X)^-1 - (X'X)^-1 R' [R(X'X)^-1 R']^-1 R (X'X)^-1
        XtX_inv = np.linalg.inv(X_clean.T @ X_clean)
        R = constraint_matrix
        RXR_inv = np.linalg.inv(R @ XtX_inv @ R.T)
        cov_beta = XtX_inv - XtX_inv @ R.T @ RXR_inv @ R @ XtX_inv
        cov_beta *= mse

        se = np.sqrt(np.diag(cov_beta))

        # Store results
        class ConstrainedResults:
            def __init__(self):
                self.params = beta
                self.bse = se
                self.resid = residuals
                self.fittedvalues = y_pred
                self.nobs = n
                self.df_resid = n - k
                self.mse_resid = mse
                self.cov_params_default = cov_beta
                self.model = type('obj', (object,), {
                    'exog': X_clean,
                    'endog': y_clean
                })

        results = ConstrainedResults()

    else:
        model = sm.OLS(y_clean, X_clean)
        results = model.fit()

    return results, data.index


def f_test_sum(results, param_indices):
    """Perform F-test for sum of coefficients = 0"""
    R = np.zeros((1, len(results.params)))
    R[0, param_indices] = 1
    q = np.array([0])

    f_stat = (R @ results.params - q).T @ np.linalg.inv(R @ results.cov_params_default @ R.T) @ (R @ results.params - q) / 1
    p_value = 1 - stats.f.cdf(f_stat, 1, results.df_resid)

    return p_value[0] if isinstance(p_value, np.ndarray) else p_value


def f_test_joint(results, param_indices):
    """Perform F-test for joint significance"""
    k = len(param_indices)
    R = np.zeros((k, len(results.params)))
    for i, idx in enumerate(param_indices):
        R[i, idx] = 1
    q = np.zeros(k)

    f_stat = (R @ results.params - q).T @ np.linalg.inv(R @ results.cov_params_default @ R.T) @ (R @ results.params - q) / k
    p_value = 1 - stats.f.cdf(f_stat, k, results.df_resid)

    return p_value


#*******************************WAGE EQUATION (gw)********************************
print("\n" + "="*80)
print("ESTIMATING WAGE EQUATION (gw)")
print("="*80)

# Prepare variables for wage equation
wage_vars = []
for i in range(1, 5):
    df[f'L{i}_gw'] = df['gw'].shift(i)
    df[f'L{i}_cf1'] = df['cf1'].shift(i)
    df[f'L{i}_vu'] = df['vu'].shift(i)
    df[f'L{i}_diffcpicf'] = df['diffcpicf'].shift(i)
    wage_vars.extend([f'L{i}_gw', f'L{i}_cf1'])

df['L1_magpty'] = df['magpty'].shift(1)

# Build X matrix for wage equation
X_wage = df[['L1_gw', 'L2_gw', 'L3_gw', 'L4_gw',
             'L1_cf1', 'L2_cf1', 'L3_cf1', 'L4_cf1',
             'L1_magpty',
             'L1_vu', 'L2_vu', 'L3_vu', 'L4_vu',
             'L1_diffcpicf', 'L2_diffcpicf', 'L3_diffcpicf', 'L4_diffcpicf',
             'dummyq2_2020', 'dummyq3_2020']].copy()

y_wage = df['gw'].copy()

# Constraint: sum of gw lags + sum of cf1 lags = 1
# Indices: 0-3 (L1-L4 gw), 4-7 (L1-L4 cf1), then constant term at position 0 after adding
constraint_R = np.zeros((1, X_wage.shape[1]))
constraint_R[0, 0:4] = 1  # L1-L4 gw
constraint_R[0, 4:8] = 1  # L1-L4 cf1
constraint_q = np.array([1.0])

print("Running constrained regression...")
results_wage, valid_idx = constrained_regression(y_wage, X_wage, constraint_R, constraint_q, add_constant=True)

# Store predictions and residuals
df.loc[valid_idx, 'gwf1'] = results_wage.fittedvalues
df['gw_residuals'] = df['gw'] - df['gwf1']

# Extract coefficients (skip constant which is first)
coef_names = ['const'] + list(X_wage.columns)
coef_df = pd.DataFrame({
    'Variable': coef_names,
    'beta': results_wage.params
})

# Save coefficients to Excel
print("Saving coefficients...")
with pd.ExcelWriter(output_dir / 'eq_coefficients_python.xlsx', engine='openpyxl', mode='w') as writer:
    coef_df.to_excel(writer, sheet_name='gw', index=False)

# Compute sum of coefficients (indices after constant)
const_offset = 1  # Position of constant in params
aa1 = sum(results_wage.params[const_offset + i] for i in range(0, 4))  # L1-L4 gw
bb1 = sum(results_wage.params[const_offset + 8 + i] for i in range(0, 4))  # L1-L4 vu
cc1 = sum(results_wage.params[const_offset + 12 + i] for i in range(0, 4))  # L1-L4 diffcpicf
dd1 = sum(results_wage.params[const_offset + 4 + i] for i in range(0, 4))  # L1-L4 cf1

print(f"\nSum of coefficients:")
print(f"  L1-L4 gw: {aa1:.7f}")
print(f"  L1-L4 vu: {bb1:.7f}")
print(f"  L1-L4 diffcpicf: {cc1:.7f}")
print(f"  L1-L4 cf1: {dd1:.7f}")

# Compute test statistics
print("Computing hypothesis tests...")
p_sum_gw = f_test_sum(results_wage, [const_offset + i for i in range(0, 4)])
p_sum_cf1 = f_test_sum(results_wage, [const_offset + 4 + i for i in range(0, 4)])
p_sum_vu = f_test_sum(results_wage, [const_offset + 9 + i for i in range(0, 4)])
p_sum_diffcpicf = f_test_sum(results_wage, [const_offset + 13 + i for i in range(0, 4)])

p_joint_gw = f_test_joint(results_wage, [const_offset + i for i in range(0, 4)])
p_joint_cf1 = f_test_joint(results_wage, [const_offset + 4 + i for i in range(0, 4)])
p_joint_vu = f_test_joint(results_wage, [const_offset + 9 + i for i in range(0, 4)])
p_joint_diffcpicf = f_test_joint(results_wage, [const_offset + 13 + i for i in range(0, 4)])
p_joint_magpty = f_test_joint(results_wage, [const_offset + 8])

# Calculate R-squared (correlation squared)
valid_data = df.dropna(subset=['gw', 'gwf1'])
valid_data = valid_data[valid_data['period'] >= '1990-01-01']
r2_wage = np.corrcoef(valid_data['gw'], valid_data['gwf1'])[0, 1] ** 2

# Create summary statistics DataFrame
summary_wage = pd.DataFrame({
    'Variable Group': [
        'l1.gw through l4.gw',
        'l1.cf1 through l4.cf1',
        'l1.vu through l4.vu',
        'l1.diffcpicf through l4.diffcpicf',
        'l1.magpty',
        '',
        'R2',
        'number of observations'
    ],
    'sum of coefficients': [aa1, dd1, bb1, cc1, results_wage.params[const_offset + 8], '', r2_wage, results_wage.nobs],
    'p value (sum)': [p_sum_gw, p_sum_cf1, p_sum_vu, p_sum_diffcpicf, '', '', '', ''],
    'p value (joint)': [p_joint_gw, p_joint_cf1, p_joint_vu, p_joint_diffcpicf, p_joint_magpty, '', '', '']
})

# Save summary statistics
with pd.ExcelWriter(output_dir / 'summary_stats_full_python.xlsx', engine='openpyxl', mode='w') as writer:
    summary_wage.to_excel(writer, sheet_name='gw', index=False)

print(f"R-squared: {r2_wage:.7f}")
print(f"Number of observations: {results_wage.nobs}")


#*******************************PRICE EQUATION (gcpi)*****************************
print("\n" + "="*80)
print("ESTIMATING PRICE EQUATION (gcpi)")
print("="*80)

# Prepare variables for price equation
for i in range(1, 5):
    df[f'L{i}_gcpi'] = df['gcpi'].shift(i)
    df[f'L{i}_grpe'] = df['grpe'].shift(i)
    df[f'L{i}_grpf'] = df['grpf'].shift(i)
    df[f'L{i}_shortage'] = df['shortage'].shift(i)

# Also need current and lagged gw
for i in range(1, 5):
    if f'L{i}_gw' not in df.columns:
        df[f'L{i}_gw'] = df['gw'].shift(i)

# Build X matrix for price equation
X_price = df[['magpty',
              'L1_gcpi', 'L2_gcpi', 'L3_gcpi', 'L4_gcpi',
              'gw', 'L1_gw', 'L2_gw', 'L3_gw', 'L4_gw',
              'grpe', 'L1_grpe', 'L2_grpe', 'L3_grpe', 'L4_grpe',
              'grpf', 'L1_grpf', 'L2_grpf', 'L3_grpf', 'L4_grpf',
              'shortage', 'L1_shortage', 'L2_shortage', 'L3_shortage', 'L4_shortage']].copy()

y_price = df['gcpi'].copy()

# Constraint: sum of gcpi lags + sum of gw (current + lags) = 1
constraint_R = np.zeros((1, X_price.shape[1]))
constraint_R[0, 1:5] = 1   # L1-L4 gcpi (indices 1-4)
constraint_R[0, 5:10] = 1  # gw, L1-L4 gw (indices 5-9)
constraint_q = np.array([1.0])

print("Running constrained regression...")
results_price, valid_idx = constrained_regression(y_price, X_price, constraint_R, constraint_q, add_constant=True)

# Store predictions and residuals
df.loc[valid_idx, 'gcpif'] = results_price.fittedvalues
df['gcpi_residuals'] = df['gcpi'] - df['gcpif']

# Extract coefficients
coef_names = ['const'] + list(X_price.columns)
coef_df = pd.DataFrame({
    'Variable': coef_names,
    'beta': results_price.params
})

# Append to Excel file
print("Saving coefficients...")
with pd.ExcelWriter(output_dir / 'eq_coefficients_python.xlsx', engine='openpyxl', mode='a') as writer:
    coef_df.to_excel(writer, sheet_name='gcpi', index=False)

# Compute sum of coefficients
const_offset = 1
aa2 = sum(results_price.params[const_offset + 1 + i] for i in range(0, 4))  # L1-L4 gcpi
bb2 = sum(results_price.params[const_offset + 5 + i] for i in range(0, 5))  # gw, L1-L4 gw
cc2 = sum(results_price.params[const_offset + 10 + i] for i in range(0, 5))  # grpe, L1-L4 grpe
dd2 = sum(results_price.params[const_offset + 15 + i] for i in range(0, 5))  # grpf, L1-L4 grpf
ee2 = sum(results_price.params[const_offset + 20 + i] for i in range(0, 5))  # shortage, L1-L4 shortage

print(f"\nSum of coefficients:")
print(f"  L1-L4 gcpi: {aa2:.7f}")
print(f"  gw through L4 gw: {bb2:.7f}")
print(f"  grpe through L4 grpe: {cc2:.7f}")
print(f"  grpf through L4 grpf: {dd2:.7f}")
print(f"  shortage through L4 shortage: {ee2:.7f}")

# Compute test statistics
print("Computing hypothesis tests...")
p_sum_gcpi = f_test_sum(results_price, [const_offset + 1 + i for i in range(0, 4)])
p_sum_gw_p = f_test_sum(results_price, [const_offset + 5 + i for i in range(0, 5)])
p_sum_grpe = f_test_sum(results_price, [const_offset + 10 + i for i in range(0, 5)])
p_sum_grpf = f_test_sum(results_price, [const_offset + 15 + i for i in range(0, 5)])
p_sum_shortage = f_test_sum(results_price, [const_offset + 20 + i for i in range(0, 5)])

p_joint_gcpi = f_test_joint(results_price, [const_offset + 1 + i for i in range(0, 4)])
p_joint_gw_p = f_test_joint(results_price, [const_offset + 5 + i for i in range(0, 5)])
p_joint_grpe = f_test_joint(results_price, [const_offset + 10 + i for i in range(0, 5)])
p_joint_grpf = f_test_joint(results_price, [const_offset + 15 + i for i in range(0, 5)])
p_joint_shortage = f_test_joint(results_price, [const_offset + 20 + i for i in range(0, 5)])
p_joint_magpty_p = f_test_joint(results_price, [const_offset])

# Calculate R-squared
valid_data = df.dropna(subset=['gcpi', 'gcpif'])
valid_data = valid_data[valid_data['period'] >= '1990-01-01']
r2_price = np.corrcoef(valid_data['gcpi'], valid_data['gcpif'])[0, 1] ** 2

# Create summary statistics DataFrame
summary_price = pd.DataFrame({
    'Variable Group': [
        'l1.gcpi through l4.gcpi',
        'gw through l4.gw',
        'grpe through l4.grpe',
        'grpf through l4.grpf',
        'shortage through l4.shortage',
        'magpty',
        '',
        'R2',
        'number of observations'
    ],
    'sum of coefficients': [aa2, bb2, cc2, dd2, ee2, results_price.params[const_offset], '', r2_price, results_price.nobs],
    'p value (sum)': [p_sum_gcpi, p_sum_gw_p, p_sum_grpe, p_sum_grpf, p_sum_shortage, '', '', '', ''],
    'p value (joint)': [p_joint_gcpi, p_joint_gw_p, p_joint_grpe, p_joint_grpf, p_joint_shortage, p_joint_magpty_p, '', '', '']
})

# Append summary statistics
with pd.ExcelWriter(output_dir / 'summary_stats_full_python.xlsx', engine='openpyxl', mode='a') as writer:
    summary_price.to_excel(writer, sheet_name='gcpi', index=False)

print(f"R-squared: {r2_price:.7f}")
print(f"Number of observations: {results_price.nobs}")


#*******************************1-YEAR EXPECTATIONS (cf1)*************************
print("\n" + "="*80)
print("ESTIMATING 1-YEAR EXPECTATIONS EQUATION (cf1)")
print("="*80)

# Prepare variables
for i in range(1, 5):
    if f'L{i}_cf1' not in df.columns:
        df[f'L{i}_cf1'] = df['cf1'].shift(i)
    if f'L{i}_cf10' not in df.columns:
        df[f'L{i}_cf10'] = df['cf10'].shift(i)
    if f'L{i}_gcpi' not in df.columns:
        df[f'L{i}_gcpi'] = df['gcpi'].shift(i)

# Build X matrix for cf1 equation
X_cf1 = df[['L1_cf1', 'L2_cf1', 'L3_cf1', 'L4_cf1',
            'cf10', 'L1_cf10', 'L2_cf10', 'L3_cf10', 'L4_cf10',
            'gcpi', 'L1_gcpi', 'L2_gcpi', 'L3_gcpi', 'L4_gcpi']].copy()

y_cf1 = df['cf1'].copy()

# Constraint: sum of all coefficients = 1 (no constant term)
constraint_R = np.ones((1, X_cf1.shape[1]))
constraint_q = np.array([1.0])

print("Running constrained regression (no constant)...")
results_cf1, valid_idx = constrained_regression(y_cf1, X_cf1, constraint_R, constraint_q, add_constant=False)

# Store predictions and residuals
df.loc[valid_idx, 'cf1f'] = results_cf1.fittedvalues
df['cf1_residuals'] = df['cf1'] - df['cf1f']

# Extract coefficients
coef_df = pd.DataFrame({
    'Variable': list(X_cf1.columns),
    'beta': results_cf1.params
})

# Append to Excel file
print("Saving coefficients...")
with pd.ExcelWriter(output_dir / 'eq_coefficients_python.xlsx', engine='openpyxl', mode='a') as writer:
    coef_df.to_excel(writer, sheet_name='cf1', index=False)

# Compute sum of coefficients (no constant offset since noconstant)
const_offset = 0
aa3 = sum(results_cf1.params[const_offset + i] for i in range(0, 4))  # L1-L4 cf1
bb3 = sum(results_cf1.params[const_offset + 4 + i] for i in range(0, 5))  # cf10, L1-L4 cf10
cc3 = sum(results_cf1.params[const_offset + 9 + i] for i in range(0, 5))  # gcpi, L1-L4 gcpi

print(f"\nSum of coefficients:")
print(f"  L1-L4 cf1: {aa3:.7f}")
print(f"  cf10 through L4 cf10: {bb3:.7f}")
print(f"  gcpi through L4 gcpi: {cc3:.7f}")

# Compute test statistics
print("Computing hypothesis tests...")
p_sum_cf1_e = f_test_sum(results_cf1, [const_offset + i for i in range(0, 4)])
p_sum_cf10 = f_test_sum(results_cf1, [const_offset + 4 + i for i in range(0, 5)])
p_sum_gcpi_e = f_test_sum(results_cf1, [const_offset + 9 + i for i in range(0, 5)])

p_joint_cf1_e = f_test_joint(results_cf1, [const_offset + i for i in range(0, 4)])
p_joint_cf10 = f_test_joint(results_cf1, [const_offset + 4 + i for i in range(0, 5)])
p_joint_gcpi_e = f_test_joint(results_cf1, [const_offset + 9 + i for i in range(0, 5)])

# Calculate R-squared
valid_data = df.dropna(subset=['cf1', 'cf1f'])
valid_data = valid_data[valid_data['period'] >= '1990-01-01']
r2_cf1 = np.corrcoef(valid_data['cf1'], valid_data['cf1f'])[0, 1] ** 2

# Create summary statistics DataFrame
summary_cf1 = pd.DataFrame({
    'Variable Group': [
        'l1.cf1 through l4.cf1',
        'cf10 through l4.cf10',
        'gcpi through l4.gcpi',
        '',
        'R2',
        'number of observations'
    ],
    'sum of coefficients': [aa3, bb3, cc3, '', r2_cf1, results_cf1.nobs],
    'p value (sum)': [p_sum_cf1_e, p_sum_cf10, p_sum_gcpi_e, '', '', ''],
    'p value (joint)': [p_joint_cf1_e, p_joint_cf10, p_joint_gcpi_e, '', '', '']
})

# Append summary statistics
with pd.ExcelWriter(output_dir / 'summary_stats_full_python.xlsx', engine='openpyxl', mode='a') as writer:
    summary_cf1.to_excel(writer, sheet_name='cf1', index=False)

print(f"R-squared: {r2_cf1:.7f}")
print(f"Number of observations: {results_cf1.nobs}")


#*******************************10-YEAR EXPECTATIONS (cf10)***********************
print("\n" + "="*80)
print("ESTIMATING 10-YEAR EXPECTATIONS EQUATION (cf10)")
print("="*80)

# Build X matrix for cf10 equation
X_cf10 = df[['L1_cf10', 'L2_cf10', 'L3_cf10', 'L4_cf10',
             'gcpi', 'L1_gcpi', 'L2_gcpi', 'L3_gcpi', 'L4_gcpi']].copy()

y_cf10 = df['cf10'].copy()

# Constraint: sum of all coefficients = 1 (no constant term)
constraint_R = np.ones((1, X_cf10.shape[1]))
constraint_q = np.array([1.0])

print("Running constrained regression (no constant)...")
results_cf10, valid_idx = constrained_regression(y_cf10, X_cf10, constraint_R, constraint_q, add_constant=False)

# Store predictions and residuals
df.loc[valid_idx, 'cf10f'] = results_cf10.fittedvalues
df['cf10_residuals'] = df['cf10f'] - df['cf10']  # Note: Stata has this reversed

# Extract coefficients
coef_df = pd.DataFrame({
    'Variable': list(X_cf10.columns),
    'beta': results_cf10.params
})

# Append to Excel file
print("Saving coefficients...")
with pd.ExcelWriter(output_dir / 'eq_coefficients_python.xlsx', engine='openpyxl', mode='a') as writer:
    coef_df.to_excel(writer, sheet_name='cf10', index=False)

# Compute sum of coefficients
const_offset = 0
aa4 = sum(results_cf10.params[const_offset + i] for i in range(0, 4))  # L1-L4 cf10
bb4 = sum(results_cf10.params[const_offset + 4 + i] for i in range(0, 5))  # gcpi, L1-L4 gcpi

print(f"\nSum of coefficients:")
print(f"  L1-L4 cf10: {aa4:.7f}")
print(f"  gcpi through L4 gcpi: {bb4:.7f}")

# Compute test statistics
print("Computing hypothesis tests...")
p_sum_cf10_e = f_test_sum(results_cf10, [const_offset + i for i in range(0, 4)])
p_sum_gcpi_e2 = f_test_sum(results_cf10, [const_offset + 4 + i for i in range(0, 5)])

p_joint_cf10_e = f_test_joint(results_cf10, [const_offset + i for i in range(0, 4)])
p_joint_gcpi_e2 = f_test_joint(results_cf10, [const_offset + 4 + i for i in range(0, 5)])

# Calculate R-squared
valid_data = df.dropna(subset=['cf10', 'cf10f'])
valid_data = valid_data[valid_data['period'] >= '1990-01-01']
r2_cf10 = np.corrcoef(valid_data['cf10'], valid_data['cf10f'])[0, 1] ** 2

# Create summary statistics DataFrame
summary_cf10 = pd.DataFrame({
    'Variable Group': [
        'l1.cf10 through l4.cf10',
        'gcpi through l4.gcpi',
        '',
        'R2',
        'number of observations'
    ],
    'sum of coefficients': [aa4, bb4, '', r2_cf10, results_cf10.nobs],
    'p value (sum)': [p_sum_cf10_e, p_sum_gcpi_e2, '', '', ''],
    'p value (joint)': [p_joint_cf10_e, p_joint_gcpi_e2, '', '', '']
})

# Append summary statistics
with pd.ExcelWriter(output_dir / 'summary_stats_full_python.xlsx', engine='openpyxl', mode='a') as writer:
    summary_cf10.to_excel(writer, sheet_name='cf10', index=False)

print(f"R-squared: {r2_cf10:.7f}")
print(f"Number of observations: {results_cf10.nobs}")


#*******************************EXPORT DATA***************************************
print("\n" + "="*80)
print("EXPORTING FINAL DATASET")
print("="*80)

# Drop temporary sum variables (they don't exist in Python version, but for consistency)
# Keep all relevant variables for export
export_vars = ['period', 'gcpi', 'vu', 'gw', 'magpty', 'grpe', 'grpf', 'cf1', 'cf10',
               'shortage', 'diffcpicf', 'CPIENGSL', 'CPIUFDSL', 'dummyq2_2020', 'dummyq3_2020',
               'gwf1', 'gw_residuals', 'gcpif', 'gcpi_residuals', 'cf1f', 'cf1_residuals',
               'cf10f', 'cf10_residuals', 'r2aa', 'n_obsaa', 'r2bb', 'n_obsbb', 'r2cc',
               'n_obscc', 'r2dd', 'n_obsdd']

# Add R2 and nobs as columns
df['r2aa'] = r2_wage
df['n_obsaa'] = results_wage.nobs
df['r2bb'] = r2_price
df['n_obsbb'] = results_price.nobs
df['r2cc'] = r2_cf1
df['n_obscc'] = results_cf1.nobs
df['r2dd'] = r2_cf10
df['n_obsdd'] = results_cf10.nobs

# Select only variables that exist
export_cols = [col for col in export_vars if col in df.columns]
df_export = df[export_cols].copy()

# Export to Excel
output_file = output_dir / 'eq_simulations_data_python.xlsx'
print(f"Exporting data to {output_file}...")
df_export.to_excel(output_file, index=False)

print("\n" + "="*80)
print("REPLICATION COMPLETE!")
print("="*80)
print(f"\nOutput files saved to: {output_dir}")
print("  - eq_coefficients_python.xlsx (4 sheets: gw, gcpi, cf1, cf10)")
print("  - summary_stats_full_python.xlsx (4 sheets: gw, gcpi, cf1, cf10)")
print("  - eq_simulations_data_python.xlsx (full dataset with predictions)")
print("\n")

# %%
