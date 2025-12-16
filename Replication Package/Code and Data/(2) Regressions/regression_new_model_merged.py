# %%
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modified Bernanke-Blanchard Model - Merged Version
Liang & Sun (2025)

This script estimates the modified model with:
1. Wage equation: adds capacity utilization
2. Shortage equation: endogenizes shortages as f(excess demand, GSCPI)
3. Price equation: same structure as BB
4. Expectations equations: same as BB

Configuration:
- Set USE_PRE_COVID_SAMPLE = True for pre-COVID estimation (out-of-sample predictions)
- Set USE_PRE_COVID_SAMPLE = False for full sample estimation
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats
from scipy.optimize import minimize
from pathlib import Path
import warnings
import seaborn as sns
warnings.filterwarnings('ignore')

# %%
#****************************CONFIGURATION**************************************

# Set to True for pre-COVID sample estimation, False for full sample
USE_PRE_COVID_SAMPLE = True

#****************************CHANGE PATH HERE***********************************

# Input Location
input_path = Path("/Users/johnathansun/Documents/ec1499/Replication Package/Code and Data/(1) Data/Public Data/Regression_Data.xlsx")

# Output Location (automatically set based on configuration)
if USE_PRE_COVID_SAMPLE:
    output_dir = Path("/Users/johnathansun/Documents/ec1499/Replication Package/Code and Data/(2) Regressions/Output Data Python (New Model Pre Covid)")
    output_suffix = "_pre_covid"
else:
    output_dir = Path("/Users/johnathansun/Documents/ec1499/Replication Package/Code and Data/(2) Regressions/Output Data Python (New Model)")
    output_suffix = ""

output_dir.mkdir(parents=True, exist_ok=True)

#*******************************************************************************

print("="*80)
if USE_PRE_COVID_SAMPLE:
    print("MODIFIED BERNANKE-BLANCHARD MODEL - PRE-COVID SAMPLE")
else:
    print("MODIFIED BERNANKE-BLANCHARD MODEL - FULL SAMPLE")
print("Liang & Sun (2025)")
print("="*80)

# %%
# LOAD DATA
print("\nLoading data...")
df = pd.read_excel(input_path)

print(f"Loaded {len(df)} observations")
print(f"Columns: {df.columns.tolist()}")

# %%
# IDENTIFY NEW VARIABLE COLUMNS
tcu_col = 'TCU'

ngdppot_col = None
for col in df.columns:
    if 'nominal potential' in col.lower() or 'ngdppot' in col.lower() or col == 'NGDPPOT':
        ngdppot_col = col
        break

gscpi_col = None
for col in df.columns:
    if 'gscpi' in col.lower() or col == 'GSCPI':
        gscpi_col = col
        break

print(f"\nIdentified new variable columns:")
print(f"  TCU column: {tcu_col}")
print(f"  NGDPPOT column: {ngdppot_col}")
print(f"  GSCPI column: {gscpi_col}")

# %%
# DATA MANIPULATION
print("\nCreating derived variables...")

# Convert Date to datetime if needed
if 'Date' in df.columns:
    if not pd.api.types.is_datetime64_any_dtype(df['Date']):
        df['Date'] = pd.to_datetime(df['Date'].apply(lambda x: x.replace(" ", "") if isinstance(x, str) else x))
    df['period'] = df['Date']
    df = df.drop('Date', axis=1)

df = df.sort_values('period').reset_index(drop=True)

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

# Other variables from BB
df['vu'] = df['VOVERU']
df['cf1'] = df['EXPINF1YR']
df['cf10'] = df['EXPINF10YR']

# Shortage (replace missing with 5)
df['shortage'] = df['SHORTAGE'].fillna(5)

# Catchup term
df['diffcpicf'] = 0.25 * sum([df['gcpi'].shift(i) for i in range(4)]) - df['cf1'].shift(4)

# COVID dummies (only used in full sample mode)
df['dummyq2_2020'] = 0.0
df['dummyq3_2020'] = 0.0
df['year'] = df['period'].dt.year
df['quarter'] = df['period'].dt.quarter
df.loc[(df['year'] == 2020) & (df['quarter'] == 2), 'dummyq2_2020'] = 1.0
df.loc[(df['year'] == 2020) & (df['quarter'] == 3), 'dummyq3_2020'] = 1.0

# %%
# NEW VARIABLES FOR MODIFIED MODEL
print("\nProcessing new model variables...")

df['tcu'] = df[tcu_col]
print(f"  TCU: {df['tcu'].notna().sum()} non-null values")

if ngdppot_col and ngdppot_col in df.columns:
    df['ngdppot'] = df[ngdppot_col]
    print(f"  NGDPPOT: {df['ngdppot'].notna().sum()} non-null values")
else:
    print("  WARNING: NGDPPOT column not found!")
    df['ngdppot'] = np.nan

if gscpi_col and gscpi_col in df.columns:
    df['gscpi'] = df[gscpi_col]
    print(f"  GSCPI: {df['gscpi'].notna().sum()} non-null values")
else:
    print("  WARNING: GSCPI column not found!")
    df['gscpi'] = np.nan

# Create log variables for new model
df['log_ngdppot'] = np.log(df['ngdppot'])
df['log_w'] = np.log(df['ECIWAG'])

# Detrend capacity utilization: log(TCU) - log(10-year rolling mean of TCU)
df['log_tcu'] = np.log(df['tcu']/100)
df['log_tcu_trend'] = np.log(df['tcu'].rolling(window=40, min_periods=20).mean()/100)
df['cu'] = df['log_tcu'] - df['log_tcu_trend']

# Excess demand proxy (uses raw log TCU, then detrend excess demand itself)
df['excess_demand'] = df['log_w'] - df['log_ngdppot'] - df['log_tcu']
df['excess_demand_trend'] = df['excess_demand'].rolling(window=40).mean()

df['excess_demand'] = df['excess_demand'] - df['excess_demand_trend']
df['excess_demand'] = (df['excess_demand'] - df['excess_demand'].mean()) / df['excess_demand'].std()

print(f"\nExcess demand proxy statistics:")
print(f"  Mean: {df['excess_demand'].mean():.4f}")
print(f"  Std:  {df['excess_demand'].std():.4f}")

# %%
# extra plotting code
# sns.lineplot(x='period', y='excess_demand', data=df, label='Excess Demand')
# sns.lineplot(x='period', y='log_w', data=df, label='Log W')
# sns.lineplot(x='period', y='log_ngdppot', data=df, label='Log NGDPPOT')
# sns.lineplot(x='period', y='cu', data=df, label='CU')
# sns.lineplot(x='period', y='cu', data=df, label='Log TCU')


# %%
# Filter to sample period: 1989 Q1 to 2023 Q2
df = df[(df['period'] >= '1989-01-01') & (df['period'] <= '2023-06-30')].copy()
df = df.reset_index(drop=True)

# Pre-COVID mask (used for wage and expectations equations in pre-COVID mode)
pre_covid_mask = df['period'] <= '2019-12-31'

print(f"\nSample size: {len(df)} observations")
print(f"Sample period: {df['period'].min()} to {df['period'].max()}")
if USE_PRE_COVID_SAMPLE:
    print(f"Pre-COVID sample: {pre_covid_mask.sum()} observations")

print(f"\nNew variable availability in sample:")
print(f"  TCU: {df['tcu'].notna().sum()}/{len(df)}")
print(f"  NGDPPOT: {df['ngdppot'].notna().sum()}/{len(df)}")
print(f"  GSCPI: {df['gscpi'].notna().sum()}/{len(df)}")

# %%
# HELPER FUNCTIONS
def constrained_regression(y, X, constraint_matrix, constraint_value, add_constant=True):
    """Perform constrained OLS regression"""
    data = pd.DataFrame(X).copy()
    data['y'] = y
    data = data.dropna()

    y_clean = data['y'].values
    X_clean = data.drop('y', axis=1).values

    if add_constant:
        X_clean = sm.add_constant(X_clean)
        if constraint_matrix is not None:
            n_constraints = constraint_matrix.shape[0]
            constraint_matrix = np.column_stack([np.zeros((n_constraints, 1)), constraint_matrix])

    if constraint_matrix is not None and constraint_value is not None:
        def objective(beta):
            resid = y_clean - X_clean @ beta
            return np.sum(resid ** 2)

        def constraint_eq(beta):
            return constraint_matrix @ beta - constraint_value

        beta_init = np.linalg.lstsq(X_clean, y_clean, rcond=None)[0]
        cons = {'type': 'eq', 'fun': constraint_eq}
        result = minimize(objective, beta_init, method='SLSQP', constraints=cons)

        beta = result.x
        y_pred = X_clean @ beta
        residuals = y_clean - y_pred

        n = len(y_clean)
        k = len(beta)
        mse = np.sum(residuals ** 2) / (n - k)

        XtX_inv = np.linalg.inv(X_clean.T @ X_clean)
        R = constraint_matrix
        RXR_inv = np.linalg.inv(R @ XtX_inv @ R.T)
        cov_beta = XtX_inv - XtX_inv @ R.T @ RXR_inv @ R @ XtX_inv
        cov_beta *= mse
        se = np.sqrt(np.diag(cov_beta))

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
                self.model = type('obj', (object,), {'exog': X_clean, 'endog': y_clean})

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


def f_test_sum_ols(results, param_indices):
    """F-test for sum of coefficients = 0, for statsmodels OLS results"""
    R = np.zeros((1, len(results.params)))
    R[0, param_indices] = 1
    f_result = results.f_test(R)
    return float(f_result.pvalue)


def f_test_joint_ols(results, param_indices):
    """F-test for joint significance, for statsmodels OLS results"""
    k = len(param_indices)
    R = np.zeros((k, len(results.params)))
    for i, idx in enumerate(param_indices):
        R[i, idx] = 1
    f_result = results.f_test(R)
    return float(f_result.pvalue)


def predict_out_of_sample(X_full, params, add_constant=True):
    """Generate predictions for full sample using estimated coefficients"""
    X_clean = X_full.values
    if add_constant:
        X_clean = sm.add_constant(X_clean)
    return X_clean @ params


# %%
#*******************************************************************************
# EQUATION 1: MODIFIED WAGE EQUATION (gw)
#*******************************************************************************
print("\n" + "="*80)
sample_label = "PRE-COVID SAMPLE" if USE_PRE_COVID_SAMPLE else "FULL SAMPLE"
print(f"EQUATION 1: MODIFIED WAGE EQUATION (gw) - {sample_label}")
print("="*80)
print("Modification: Adding capacity utilization terms")
if USE_PRE_COVID_SAMPLE:
    print("Note: NO dummy variables for COVID periods")

# Create lagged variables
for i in range(1, 5):
    df[f'L{i}_gw'] = df['gw'].shift(i)
    df[f'L{i}_cf1'] = df['cf1'].shift(i)
    df[f'L{i}_vu'] = df['vu'].shift(i)
    df[f'L{i}_diffcpicf'] = df['diffcpicf'].shift(i)
    df[f'L{i}_cu'] = df['cu'].shift(i)

df['L1_magpty'] = df['magpty'].shift(1)

# Build X matrix for wage equation - includes capacity utilization
X_wage_cols = ['L1_gw', 'L2_gw', 'L3_gw', 'L4_gw',
               'L1_cf1', 'L2_cf1', 'L3_cf1', 'L4_cf1',
               'L1_magpty',
               'L1_vu', 'L2_vu', 'L3_vu', 'L4_vu',
               'L1_diffcpicf', 'L2_diffcpicf', 'L3_diffcpicf', 'L4_diffcpicf',
               'L1_cu', 'L2_cu', 'L3_cu', 'L4_cu']

# Add COVID dummies only in full sample mode
if not USE_PRE_COVID_SAMPLE:
    X_wage_cols += ['dummyq2_2020', 'dummyq3_2020']

X_wage_full = df[X_wage_cols].copy()
y_wage_full = df['gw'].copy()

# Filter to appropriate sample for estimation
if USE_PRE_COVID_SAMPLE:
    X_wage = X_wage_full[pre_covid_mask].copy()
    y_wage = y_wage_full[pre_covid_mask].copy()
else:
    X_wage = X_wage_full.copy()
    y_wage = y_wage_full.copy()

# Constraint: sum of gw lags + sum of cf1 lags = 1
constraint_R = np.zeros((1, len(X_wage_cols)))
constraint_R[0, 0:4] = 1   # L1-L4 gw
constraint_R[0, 4:8] = 1   # L1-L4 cf1
constraint_q = np.array([1.0])

print(f"Running constrained regression on {sample_label.lower()}...")
results_wage, valid_idx = constrained_regression(y_wage, X_wage, constraint_R, constraint_q, add_constant=True)

# Generate predictions
if USE_PRE_COVID_SAMPLE:
    print("Generating out-of-sample predictions...")
    df['gwf1'] = predict_out_of_sample(X_wage_full, results_wage.params, add_constant=True)
else:
    df.loc[valid_idx, 'gwf1'] = results_wage.fittedvalues

df['gw_residuals'] = df['gw'] - df['gwf1']

# Extract coefficients
coef_names = ['const'] + list(X_wage_cols)
coef_df = pd.DataFrame({'Variable': coef_names, 'beta': results_wage.params, 'se': results_wage.bse})

# Save coefficients
with pd.ExcelWriter(output_dir / f'eq_coefficients_new_model{output_suffix}.xlsx', engine='openpyxl', mode='w') as writer:
    coef_df.to_excel(writer, sheet_name='gw', index=False)

# Compute sum of coefficients
const_offset = 1
sum_gw = sum(results_wage.params[const_offset + i] for i in range(0, 4))
sum_cf1 = sum(results_wage.params[const_offset + 4 + i] for i in range(0, 4))
sum_vu = sum(results_wage.params[const_offset + 9 + i] for i in range(0, 4))
sum_diffcpicf = sum(results_wage.params[const_offset + 13 + i] for i in range(0, 4))
sum_cu = sum(results_wage.params[const_offset + 17 + i] for i in range(0, 4))

print(f"\nSum of coefficients:")
print(f"  L1-L4 gw:       {sum_gw:.6f}")
print(f"  L1-L4 cf1:      {sum_cf1:.6f}")
print(f"  L1-L4 vu:       {sum_vu:.6f}")
print(f"  L1-L4 diffcpicf:{sum_diffcpicf:.6f}")
print(f"  L1-L4 cu:       {sum_cu:.6f} [NEW]")

# Test significance of capacity utilization
p_sum_cu = f_test_sum(results_wage, [const_offset + 17 + i for i in range(0, 4)])
p_joint_cu = f_test_joint(results_wage, [const_offset + 17 + i for i in range(0, 4)])
print(f"\nCapacity utilization significance:")
print(f"  Sum = 0 test p-value:   {p_sum_cu:.6f}")
print(f"  Joint test p-value:     {p_joint_cu:.6f}")

# R-squared
valid_data = df.dropna(subset=['gw', 'gwf1'])
if USE_PRE_COVID_SAMPLE:
    valid_data = valid_data[(valid_data['period'] >= '1990-01-01') & (valid_data['period'] <= '2019-12-31')]
else:
    valid_data = valid_data[valid_data['period'] >= '1990-01-01']
r2_wage = np.corrcoef(valid_data['gw'], valid_data['gwf1'])[0, 1] ** 2
r2_label = "R-squared (pre-COVID)" if USE_PRE_COVID_SAMPLE else "R-squared"
print(f"\n{r2_label}: {r2_wage:.6f}")
print(f"Number of observations: {results_wage.nobs}")


# %%
#*******************************************************************************
# EQUATION 2: SHORTAGE EQUATION (NEW) - ALWAYS FULL SAMPLE
#*******************************************************************************
print("\n" + "="*80)
print("EQUATION 2: SHORTAGE EQUATION (NEW) - FULL SAMPLE")
print("="*80)
print("shortage = f(lagged shortage, excess demand, GSCPI)")
if USE_PRE_COVID_SAMPLE:
    print("Note: Estimated on full sample because shortage data is most informative during COVID")

# Create lagged shortage
for i in range(1, 5):
    df[f'L{i}_shortage'] = df['shortage'].shift(i)
    df[f'L{i}_excess_demand'] = df['excess_demand'].shift(i)
    df[f'L{i}_gscpi'] = df['gscpi'].shift(i)

# Build X matrix for shortage equation
X_shortage_cols = ['L1_shortage', 'L2_shortage', 'L3_shortage', 'L4_shortage',
                   'excess_demand', 'L1_excess_demand', 'L2_excess_demand', 'L3_excess_demand', 'L4_excess_demand',
                   'gscpi', 'L1_gscpi', 'L2_gscpi', 'L3_gscpi', 'L4_gscpi']

X_shortage = df[X_shortage_cols].copy()
y_shortage = df['shortage'].copy()

# Align indices
print("Running OLS regression on FULL sample...")
shortage_data = pd.concat([y_shortage, X_shortage], axis=1).dropna()
y_short_clean = shortage_data['shortage']
X_short_clean = shortage_data.drop('shortage', axis=1)

model_shortage = sm.OLS(y_short_clean, sm.add_constant(X_short_clean))
results_shortage = model_shortage.fit()

# Store predictions
df.loc[shortage_data.index, 'shortagef'] = results_shortage.fittedvalues
df['shortage_residuals'] = df['shortage'] - df['shortagef']

# Extract coefficients
coef_names = ['const'] + list(X_shortage_cols)
coef_df = pd.DataFrame({
    'Variable': coef_names,
    'beta': results_shortage.params,
    'se': results_shortage.bse,
    't_stat': results_shortage.tvalues,
    'p_value': results_shortage.pvalues
})

with pd.ExcelWriter(output_dir / f'eq_coefficients_new_model{output_suffix}.xlsx', engine='openpyxl', mode='a') as writer:
    coef_df.to_excel(writer, sheet_name='shortage', index=False)

# Sum of coefficients
sum_shortage_lag = sum(results_shortage.params[1:5])
sum_excess_demand = sum(results_shortage.params[5:10])
sum_gscpi = sum(results_shortage.params[10:15])

print(f"\nSum of coefficients:")
print(f"  L1-L4 shortage:     {sum_shortage_lag:.6f} (persistence)")
print(f"  excess_demand terms:{sum_excess_demand:.6f} (demand effect)")
print(f"  GSCPI terms:        {sum_gscpi:.6f} (supply chain effect)")

# Long-run multipliers
if abs(1 - sum_shortage_lag) > 0.01:
    lr_excess_demand = sum_excess_demand / (1 - sum_shortage_lag)
    lr_gscpi = sum_gscpi / (1 - sum_shortage_lag)
    print(f"\nLong-run multipliers:")
    print(f"  Excess demand -> shortage: {lr_excess_demand:.4f}")
    print(f"  GSCPI -> shortage:         {lr_gscpi:.4f}")
else:
    lr_excess_demand = np.nan
    lr_gscpi = np.nan

print(f"\nR-squared: {results_shortage.rsquared:.6f}")
print(f"Number of observations: {results_shortage.nobs}")


# %%
#*******************************************************************************
# EQUATION 3: PRICE EQUATION (gcpi) - ALWAYS FULL SAMPLE
#*******************************************************************************
print("\n" + "="*80)
print("EQUATION 3: PRICE EQUATION (gcpi) - FULL SAMPLE")
print("="*80)
print("Same structure as Bernanke-Blanchard")

# Create lagged variables
for i in range(1, 5):
    if f'L{i}_gcpi' not in df.columns:
        df[f'L{i}_gcpi'] = df['gcpi'].shift(i)
    if f'L{i}_grpe' not in df.columns:
        df[f'L{i}_grpe'] = df['grpe'].shift(i)
    if f'L{i}_grpf' not in df.columns:
        df[f'L{i}_grpf'] = df['grpf'].shift(i)

# Build X matrix
X_price_cols = ['magpty',
                'L1_gcpi', 'L2_gcpi', 'L3_gcpi', 'L4_gcpi',
                'gw', 'L1_gw', 'L2_gw', 'L3_gw', 'L4_gw',
                'grpe', 'L1_grpe', 'L2_grpe', 'L3_grpe', 'L4_grpe',
                'grpf', 'L1_grpf', 'L2_grpf', 'L3_grpf', 'L4_grpf',
                'shortage', 'L1_shortage', 'L2_shortage', 'L3_shortage', 'L4_shortage']

X_price = df[X_price_cols].copy()
y_price = df['gcpi'].copy()

# Constraint: sum of gcpi lags + sum of gw = 1
constraint_R = np.zeros((1, len(X_price_cols)))
constraint_R[0, 1:5] = 1   # L1-L4 gcpi
constraint_R[0, 5:10] = 1  # gw, L1-L4 gw
constraint_q = np.array([1.0])

print("Running constrained regression on FULL sample...")
results_price, valid_idx = constrained_regression(y_price, X_price, constraint_R, constraint_q, add_constant=True)

df.loc[valid_idx, 'gcpif'] = results_price.fittedvalues
df['gcpi_residuals'] = df['gcpi'] - df['gcpif']

# Extract coefficients
coef_names = ['const'] + list(X_price_cols)
coef_df = pd.DataFrame({'Variable': coef_names, 'beta': results_price.params, 'se': results_price.bse})

with pd.ExcelWriter(output_dir / f'eq_coefficients_new_model{output_suffix}.xlsx', engine='openpyxl', mode='a') as writer:
    coef_df.to_excel(writer, sheet_name='gcpi', index=False)

# Sum of coefficients
const_offset_p = 1
sum_gcpi = sum(results_price.params[const_offset_p + 1 + i] for i in range(0, 4))
sum_gw_price = sum(results_price.params[const_offset_p + 5 + i] for i in range(0, 5))
sum_grpe = sum(results_price.params[const_offset_p + 10 + i] for i in range(0, 5))
sum_grpf = sum(results_price.params[const_offset_p + 15 + i] for i in range(0, 5))
sum_shortage_price = sum(results_price.params[const_offset_p + 20 + i] for i in range(0, 5))

print(f"\nSum of coefficients:")
print(f"  L1-L4 gcpi:    {sum_gcpi:.6f}")
print(f"  gw terms:      {sum_gw_price:.6f}")
print(f"  grpe terms:    {sum_grpe:.6f}")
print(f"  grpf terms:    {sum_grpf:.6f}")
print(f"  shortage terms:{sum_shortage_price:.6f}")

valid_data = df.dropna(subset=['gcpi', 'gcpif'])
valid_data = valid_data[valid_data['period'] >= '1990-01-01']
r2_price = np.corrcoef(valid_data['gcpi'], valid_data['gcpif'])[0, 1] ** 2
print(f"\nR-squared: {r2_price:.6f}")
print(f"Number of observations: {results_price.nobs}")


# %%
#*******************************************************************************
# EQUATION 4: 1-YEAR EXPECTATIONS (cf1)
#*******************************************************************************
print("\n" + "="*80)
sample_label = "PRE-COVID SAMPLE" if USE_PRE_COVID_SAMPLE else "FULL SAMPLE"
print(f"EQUATION 4: 1-YEAR EXPECTATIONS (cf1) - {sample_label}")
print("="*80)

# Create lagged variables
for i in range(1, 5):
    if f'L{i}_cf10' not in df.columns:
        df[f'L{i}_cf10'] = df['cf10'].shift(i)

X_cf1_cols = ['L1_cf1', 'L2_cf1', 'L3_cf1', 'L4_cf1',
              'cf10', 'L1_cf10', 'L2_cf10', 'L3_cf10', 'L4_cf10',
              'gcpi', 'L1_gcpi', 'L2_gcpi', 'L3_gcpi', 'L4_gcpi']

X_cf1_full = df[X_cf1_cols].copy()
y_cf1_full = df['cf1'].copy()

# Filter to appropriate sample for estimation
if USE_PRE_COVID_SAMPLE:
    X_cf1 = X_cf1_full[pre_covid_mask].copy()
    y_cf1 = y_cf1_full[pre_covid_mask].copy()
else:
    X_cf1 = X_cf1_full.copy()
    y_cf1 = y_cf1_full.copy()

# Constraint: sum of all = 1 (no constant)
constraint_R = np.ones((1, len(X_cf1_cols)))
constraint_q = np.array([1.0])

print(f"Running constrained regression (no constant) on {sample_label.lower()}...")
results_cf1, valid_idx = constrained_regression(y_cf1, X_cf1, constraint_R, constraint_q, add_constant=False)

# Generate predictions
if USE_PRE_COVID_SAMPLE:
    print("Generating out-of-sample predictions...")
    df['cf1f'] = predict_out_of_sample(X_cf1_full, results_cf1.params, add_constant=False)
else:
    df.loc[valid_idx, 'cf1f'] = results_cf1.fittedvalues

df['cf1_residuals'] = df['cf1'] - df['cf1f']

coef_df = pd.DataFrame({'Variable': list(X_cf1_cols), 'beta': results_cf1.params, 'se': results_cf1.bse})

with pd.ExcelWriter(output_dir / f'eq_coefficients_new_model{output_suffix}.xlsx', engine='openpyxl', mode='a') as writer:
    coef_df.to_excel(writer, sheet_name='cf1', index=False)

sum_cf1_lag = sum(results_cf1.params[0:4])
sum_cf10 = sum(results_cf1.params[4:9])
sum_gcpi_cf1 = sum(results_cf1.params[9:14])

print(f"\nSum of coefficients:")
print(f"  L1-L4 cf1:  {sum_cf1_lag:.6f}")
print(f"  cf10 terms: {sum_cf10:.6f}")
print(f"  gcpi terms: {sum_gcpi_cf1:.6f}")

valid_data = df.dropna(subset=['cf1', 'cf1f'])
if USE_PRE_COVID_SAMPLE:
    valid_data = valid_data[(valid_data['period'] >= '1990-01-01') & (valid_data['period'] <= '2019-12-31')]
else:
    valid_data = valid_data[valid_data['period'] >= '1990-01-01']
r2_cf1 = np.corrcoef(valid_data['cf1'], valid_data['cf1f'])[0, 1] ** 2
r2_label = "R-squared (pre-COVID)" if USE_PRE_COVID_SAMPLE else "R-squared"
print(f"\n{r2_label}: {r2_cf1:.6f}")
print(f"Number of observations: {results_cf1.nobs}")


# %%
#*******************************************************************************
# EQUATION 5: 10-YEAR EXPECTATIONS (cf10)
#*******************************************************************************
print("\n" + "="*80)
sample_label = "PRE-COVID SAMPLE" if USE_PRE_COVID_SAMPLE else "FULL SAMPLE"
print(f"EQUATION 5: 10-YEAR EXPECTATIONS (cf10) - {sample_label}")
print("="*80)

X_cf10_cols = ['L1_cf10', 'L2_cf10', 'L3_cf10', 'L4_cf10',
               'gcpi', 'L1_gcpi', 'L2_gcpi', 'L3_gcpi', 'L4_gcpi']

X_cf10_full = df[X_cf10_cols].copy()
y_cf10_full = df['cf10'].copy()

# Filter to appropriate sample for estimation
if USE_PRE_COVID_SAMPLE:
    X_cf10 = X_cf10_full[pre_covid_mask].copy()
    y_cf10 = y_cf10_full[pre_covid_mask].copy()
else:
    X_cf10 = X_cf10_full.copy()
    y_cf10 = y_cf10_full.copy()

constraint_R = np.ones((1, len(X_cf10_cols)))
constraint_q = np.array([1.0])

print(f"Running constrained regression (no constant) on {sample_label.lower()}...")
results_cf10, valid_idx = constrained_regression(y_cf10, X_cf10, constraint_R, constraint_q, add_constant=False)

# Generate predictions
if USE_PRE_COVID_SAMPLE:
    print("Generating out-of-sample predictions...")
    df['cf10f'] = predict_out_of_sample(X_cf10_full, results_cf10.params, add_constant=False)
else:
    df.loc[valid_idx, 'cf10f'] = results_cf10.fittedvalues

df['cf10_residuals'] = df['cf10'] - df['cf10f']

coef_df = pd.DataFrame({'Variable': list(X_cf10_cols), 'beta': results_cf10.params, 'se': results_cf10.bse})

with pd.ExcelWriter(output_dir / f'eq_coefficients_new_model{output_suffix}.xlsx', engine='openpyxl', mode='a') as writer:
    coef_df.to_excel(writer, sheet_name='cf10', index=False)

sum_cf10_lag = sum(results_cf10.params[0:4])
sum_gcpi_cf10 = sum(results_cf10.params[4:9])

print(f"\nSum of coefficients:")
print(f"  L1-L4 cf10: {sum_cf10_lag:.6f}")
print(f"  gcpi terms: {sum_gcpi_cf10:.6f}")

valid_data = df.dropna(subset=['cf10', 'cf10f'])
if USE_PRE_COVID_SAMPLE:
    valid_data = valid_data[(valid_data['period'] >= '1990-01-01') & (valid_data['period'] <= '2019-12-31')]
else:
    valid_data = valid_data[valid_data['period'] >= '1990-01-01']
r2_cf10 = np.corrcoef(valid_data['cf10'], valid_data['cf10f'])[0, 1] ** 2
r2_label = "R-squared (pre-COVID)" if USE_PRE_COVID_SAMPLE else "R-squared"
print(f"\n{r2_label}: {r2_cf10:.6f}")
print(f"Number of observations: {results_cf10.nobs}")


# %%
#*******************************************************************************
# EXPORT DATA
#*******************************************************************************
print("\n" + "="*80)
print("EXPORTING FINAL DATASET")
print("="*80)

# Add R2 values
df['r2_wage'] = r2_wage
df['r2_shortage'] = results_shortage.rsquared
df['r2_price'] = r2_price
df['r2_cf1'] = r2_cf1
df['r2_cf10'] = r2_cf10

# Select export columns
export_vars = ['period', 'gcpi', 'vu', 'gw', 'magpty', 'grpe', 'grpf', 'cf1', 'cf10',
               'shortage', 'diffcpicf']

# Add dummies only if in full sample mode
if not USE_PRE_COVID_SAMPLE:
    export_vars += ['dummyq2_2020', 'dummyq3_2020']

export_vars += [
    # New variables
    'tcu', 'cu', 'ngdppot', 'gscpi', 'log_ngdppot', 'excess_demand',
    # Fitted values
    'gwf1', 'shortagef', 'gcpif', 'cf1f', 'cf10f',
    # Residuals
    'gw_residuals', 'shortage_residuals', 'gcpi_residuals', 'cf1_residuals', 'cf10_residuals',
    # R2
    'r2_wage', 'r2_shortage', 'r2_price', 'r2_cf1', 'r2_cf10']

export_cols = [col for col in export_vars if col in df.columns]
df_export = df[export_cols].copy()

output_file = output_dir / f'eq_simulations_data_new_model{output_suffix}.xlsx'
df_export.to_excel(output_file, index=False)
print(f"Exported data to {output_file}")


# %%
#*******************************************************************************
# SUMMARY STATISTICS
#*******************************************************************************
print("\n" + "="*80)
print("CREATING SUMMARY STATISTICS")
print("="*80)

# Compute p-values for wage equation
p_sum_gw = f_test_sum(results_wage, [const_offset + i for i in range(0, 4)])
p_sum_cf1_wage = f_test_sum(results_wage, [const_offset + 4 + i for i in range(0, 4)])
p_sum_vu = f_test_sum(results_wage, [const_offset + 9 + i for i in range(0, 4)])
p_sum_diffcpicf = f_test_sum(results_wage, [const_offset + 13 + i for i in range(0, 4)])

p_joint_gw = f_test_joint(results_wage, [const_offset + i for i in range(0, 4)])
p_joint_cf1_wage = f_test_joint(results_wage, [const_offset + 4 + i for i in range(0, 4)])
p_joint_vu = f_test_joint(results_wage, [const_offset + 9 + i for i in range(0, 4)])
p_joint_diffcpicf = f_test_joint(results_wage, [const_offset + 13 + i for i in range(0, 4)])
p_joint_magpty = f_test_joint(results_wage, [const_offset + 8])

# Price equation p-values
p_sum_gcpi_lag = f_test_sum(results_price, [const_offset_p + 1 + i for i in range(0, 4)])
p_sum_gw_price = f_test_sum(results_price, [const_offset_p + 5 + i for i in range(0, 5)])
p_sum_grpe = f_test_sum(results_price, [const_offset_p + 10 + i for i in range(0, 5)])
p_sum_grpf = f_test_sum(results_price, [const_offset_p + 15 + i for i in range(0, 5)])
p_sum_shortage_p = f_test_sum(results_price, [const_offset_p + 20 + i for i in range(0, 5)])

p_joint_gcpi_lag = f_test_joint(results_price, [const_offset_p + 1 + i for i in range(0, 4)])
p_joint_gw_price = f_test_joint(results_price, [const_offset_p + 5 + i for i in range(0, 5)])
p_joint_grpe = f_test_joint(results_price, [const_offset_p + 10 + i for i in range(0, 5)])
p_joint_grpf = f_test_joint(results_price, [const_offset_p + 15 + i for i in range(0, 5)])
p_joint_shortage_p = f_test_joint(results_price, [const_offset_p + 20 + i for i in range(0, 5)])
p_joint_magpty_p = f_test_joint(results_price, [const_offset_p])

# Shortage equation p-values
p_sum_shortage_lag = f_test_sum_ols(results_shortage, [1, 2, 3, 4])
p_sum_excess_demand = f_test_sum_ols(results_shortage, [5, 6, 7, 8, 9])
p_sum_gscpi_short = f_test_sum_ols(results_shortage, [10, 11, 12, 13, 14])

p_joint_shortage_lag = f_test_joint_ols(results_shortage, [1, 2, 3, 4])
p_joint_excess_demand = f_test_joint_ols(results_shortage, [5, 6, 7, 8, 9])
p_joint_gscpi_short = f_test_joint_ols(results_shortage, [10, 11, 12, 13, 14])

# CF1 and CF10 p-values
p_sum_cf1_lag = f_test_sum(results_cf1, [i for i in range(0, 4)])
p_sum_cf10_cf1 = f_test_sum(results_cf1, [4 + i for i in range(0, 5)])
p_sum_gcpi_cf1 = f_test_sum(results_cf1, [9 + i for i in range(0, 5)])

p_joint_cf1_lag = f_test_joint(results_cf1, [i for i in range(0, 4)])
p_joint_cf10_cf1 = f_test_joint(results_cf1, [4 + i for i in range(0, 5)])
p_joint_gcpi_cf1 = f_test_joint(results_cf1, [9 + i for i in range(0, 5)])

p_sum_cf10_lag = f_test_sum(results_cf10, [i for i in range(0, 4)])
p_sum_gcpi_cf10 = f_test_sum(results_cf10, [4 + i for i in range(0, 5)])

p_joint_cf10_lag = f_test_joint(results_cf10, [i for i in range(0, 4)])
p_joint_gcpi_cf10 = f_test_joint(results_cf10, [4 + i for i in range(0, 5)])

# Create summary DataFrames
r2_label = 'R2 (pre-COVID)' if USE_PRE_COVID_SAMPLE else 'R2'

summary_wage = pd.DataFrame({
    'Variable Group': [
        'l1.gw through l4.gw',
        'l1.cf1 through l4.cf1',
        'l1.vu through l4.vu',
        'l1.diffcpicf through l4.diffcpicf',
        'l1.cu through l4.cu [NEW]',
        'l1.magpty',
        '',
        r2_label,
        'number of observations'
    ],
    'sum of coefficients': [sum_gw, sum_cf1, sum_vu, sum_diffcpicf, sum_cu,
                            results_wage.params[const_offset + 8], '', r2_wage, results_wage.nobs],
    'p value (sum)': [p_sum_gw, p_sum_cf1_wage, p_sum_vu, p_sum_diffcpicf, p_sum_cu, '', '', '', ''],
    'p value (joint)': [p_joint_gw, p_joint_cf1_wage, p_joint_vu, p_joint_diffcpicf, p_joint_cu, p_joint_magpty, '', '', '']
})

summary_shortage = pd.DataFrame({
    'Variable Group': [
        'l1.shortage through l4.shortage',
        'excess_demand through l4.excess_demand',
        'gscpi through l4.gscpi',
        '',
        'Long-run excess demand multiplier',
        'Long-run GSCPI multiplier',
        '',
        'R2',
        'number of observations'
    ],
    'sum of coefficients': [
        sum_shortage_lag, sum_excess_demand, sum_gscpi, '',
        lr_excess_demand if not np.isnan(lr_excess_demand) else 'N/A',
        lr_gscpi if not np.isnan(lr_gscpi) else 'N/A',
        '', results_shortage.rsquared, results_shortage.nobs
    ],
    'p value (sum)': [p_sum_shortage_lag, p_sum_excess_demand, p_sum_gscpi_short, '', '', '', '', '', ''],
    'p value (joint)': [p_joint_shortage_lag, p_joint_excess_demand, p_joint_gscpi_short, '', '', '', '', '', '']
})

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
    'sum of coefficients': [sum_gcpi, sum_gw_price, sum_grpe, sum_grpf, sum_shortage_price,
                            results_price.params[const_offset_p], '', r2_price, results_price.nobs],
    'p value (sum)': [p_sum_gcpi_lag, p_sum_gw_price, p_sum_grpe, p_sum_grpf, p_sum_shortage_p, '', '', '', ''],
    'p value (joint)': [p_joint_gcpi_lag, p_joint_gw_price, p_joint_grpe, p_joint_grpf, p_joint_shortage_p, p_joint_magpty_p, '', '', '']
})

summary_cf1 = pd.DataFrame({
    'Variable Group': [
        'l1.cf1 through l4.cf1',
        'cf10 through l4.cf10',
        'gcpi through l4.gcpi',
        '',
        r2_label,
        'number of observations'
    ],
    'sum of coefficients': [sum_cf1_lag, sum_cf10, sum_gcpi_cf1, '', r2_cf1, results_cf1.nobs],
    'p value (sum)': [p_sum_cf1_lag, p_sum_cf10_cf1, p_sum_gcpi_cf1, '', '', ''],
    'p value (joint)': [p_joint_cf1_lag, p_joint_cf10_cf1, p_joint_gcpi_cf1, '', '', '']
})

summary_cf10 = pd.DataFrame({
    'Variable Group': [
        'l1.cf10 through l4.cf10',
        'gcpi through l4.gcpi',
        '',
        r2_label,
        'number of observations'
    ],
    'sum of coefficients': [sum_cf10_lag, sum_gcpi_cf10, '', r2_cf10, results_cf10.nobs],
    'p value (sum)': [p_sum_cf10_lag, p_sum_gcpi_cf10, '', '', ''],
    'p value (joint)': [p_joint_cf10_lag, p_joint_gcpi_cf10, '', '', '']
})

# Save summary statistics
summary_file = output_dir / f'summary_stats_new_model{output_suffix}.xlsx'
with pd.ExcelWriter(summary_file, engine='openpyxl', mode='w') as writer:
    summary_wage.to_excel(writer, sheet_name='gw', index=False)
    summary_shortage.to_excel(writer, sheet_name='shortage', index=False)
    summary_price.to_excel(writer, sheet_name='gcpi', index=False)
    summary_cf1.to_excel(writer, sheet_name='cf1', index=False)
    summary_cf10.to_excel(writer, sheet_name='cf10', index=False)

print(f"Saved summary statistics to {summary_file}")


# %%
# SUMMARY
print("\n" + "="*80)
print("MODEL ESTIMATION COMPLETE!")
print("="*80)

print("\n" + "-"*60)
print("SUMMARY OF RESULTS")
print("-"*60)

sample_note = " (Pre-COVID)" if USE_PRE_COVID_SAMPLE else ""

print(f"\n1. WAGE EQUATION (Modified{sample_note})")
print(f"   R²{sample_note}: {r2_wage:.4f}")
print(f"   Capacity utilization effect (sum): {sum_cu:.4f}")
print(f"   Capacity utilization p-value (joint): {p_joint_cu:.4f}")

print(f"\n2. SHORTAGE EQUATION (New, Full Sample)")
print(f"   R²: {results_shortage.rsquared:.4f}")
print(f"   Persistence (sum of lags): {sum_shortage_lag:.4f}")
print(f"   Excess demand effect: {sum_excess_demand:.4f}")
print(f"   GSCPI effect: {sum_gscpi:.4f}")

print(f"\n3. PRICE EQUATION (Full Sample)")
print(f"   R²: {r2_price:.4f}")
print(f"   Shortage effect: {sum_shortage_price:.4f}")

print(f"\n4. SHORT-RUN EXPECTATIONS{sample_note}")
print(f"   R²{sample_note}: {r2_cf1:.4f}")

print(f"\n5. LONG-RUN EXPECTATIONS{sample_note}")
print(f"   R²{sample_note}: {r2_cf10:.4f}")

print("\n" + "-"*60)
print("KEY FINDING: SHORTAGE DECOMPOSITION")
print("-"*60)
if not np.isnan(lr_excess_demand):
    print(f"\nShortage persistence: {sum_shortage_lag:.2%}")
    print(f"Long-run excess demand multiplier: {lr_excess_demand:.4f}")
    print(f"Long-run GSCPI multiplier: {lr_gscpi:.4f}")

    # Interpret
    total_lr = abs(lr_excess_demand) + abs(lr_gscpi)
    if total_lr > 0:
        pct_demand = abs(lr_excess_demand) / total_lr * 100
        pct_supply = abs(lr_gscpi) / total_lr * 100
        print(f"\nContribution to shortages:")
        print(f"  Excess demand (wages/capacity): {pct_demand:.1f}%")
        print(f"  Supply chain pressure (GSCPI):  {pct_supply:.1f}%")

if USE_PRE_COVID_SAMPLE:
    print("\n" + "-"*60)
    print("SAMPLE CONFIGURATION:")
    print("-"*60)
    print("  - gw: estimated on pre-COVID sample, NO dummy variables")
    print("  - shortage: estimated on FULL sample (COVID data needed)")
    print("  - gcpi: estimated on FULL sample (shortage variable)")
    print("  - cf1, cf10: estimated on pre-COVID sample")
    print("  - Out-of-sample predictions for 2020+ periods")

print("\n" + "="*80)
print(f"Output files saved to: {output_dir}")
print(f"  - eq_coefficients_new_model{output_suffix}.xlsx (5 sheets)")
print(f"  - eq_simulations_data_new_model{output_suffix}.xlsx (data with predictions)")
print(f"  - summary_stats_new_model{output_suffix}.xlsx (5 sheets)")
print("="*80)

# %%
