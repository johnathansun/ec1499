# %%
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modified Bernanke-Blanchard Model - Refactored Version
Liang & Sun (2025)

This script estimates the modified model with declarative variable definitions.
Equations and variable groups are defined in a configuration dict, and
coefficient sums/p-values are computed automatically.
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats as sp_stats
from scipy.optimize import minimize
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# %%
#****************************CONFIGURATION**************************************

USE_PRE_COVID_SAMPLE = True
USE_LOG_CU_WAGES = False
USE_CONTEMP_CU = False
USE_DETRENDED_EXCESS_DEMAND = True

#****************************CHANGE PATH HERE***********************************

input_path = Path("/Users/johnathansun/Documents/ec1499/Replication Package/Code and Data/(1) Data/Public Data/Regression_Data.xlsx")

# Build output directory name based on configuration
base_output = "/Users/johnathansun/Documents/ec1499/Replication Package/Code and Data/(2) Regressions"
dir_parts = ["Output Data (New"]
if USE_PRE_COVID_SAMPLE:
    dir_parts.append("Pre Covid")
if USE_LOG_CU_WAGES:
    dir_parts.append("Log CU")
if USE_CONTEMP_CU:
    dir_parts.append("Contemp CU")
if USE_DETRENDED_EXCESS_DEMAND:
    dir_parts.append("Detrended ED")
dir_name = " ".join(dir_parts) + ")"
output_dir = Path(base_output) / dir_name
output_suffix = "_pre_covid" if USE_PRE_COVID_SAMPLE else ""

output_dir.mkdir(parents=True, exist_ok=True)

#****************************EQUATION DEFINITIONS*******************************
# Each variable group is defined as: (base_name, lags)
# - lags is a list of lag indices (0 = contemporaneous, 1 = L1, etc.)
# - Groups are automatically summed and tested together

# Wage equation variables
WAGE_VARS = [
    ('gw', [1, 2, 3, 4]),
    ('cf1', [1, 2, 3, 4]),
    ('magpty', [1]),
    ('vu', [1, 2, 3, 4]),
    ('diffcpicf', [1, 2, 3, 4]),
    ('cu', [0, 1, 2, 3, 4] if USE_CONTEMP_CU else [1, 2, 3, 4]),
]

# Shortage equation variables
SHORTAGE_VARS = [
    ('shortage', [1, 2, 3, 4]),
    ('excess_demand', [0, 1, 2, 3, 4]),
    ('gscpi', [0, 1, 2, 3, 4]),
]

# Price equation variables
PRICE_VARS = [
    ('magpty', [0]),
    ('gcpi', [1, 2, 3, 4]),
    ('gw', [0, 1, 2, 3, 4]),
    ('grpe', [0, 1, 2, 3, 4]),
    ('grpf', [0, 1, 2, 3, 4]),
    ('shortage', [0, 1, 2, 3, 4]),
]

# CF1 equation variables
CF1_VARS = [
    ('cf1', [1, 2, 3, 4]),
    ('cf10', [0, 1, 2, 3, 4]),
    ('gcpi', [0, 1, 2, 3, 4]),
]

# CF10 equation variables
CF10_VARS = [
    ('cf10', [1, 2, 3, 4]),
    ('gcpi', [0, 1, 2, 3, 4]),
]

#*******************************************************************************
# HELPER FUNCTIONS
#*******************************************************************************

def make_col_name(base, lag):
    """Generate column name for a variable at a given lag."""
    return base if lag == 0 else f'L{lag}_{base}'

def make_group_label(base, lags):
    """Generate a human-readable label for a variable group."""
    if len(lags) == 1:
        return make_col_name(base, lags[0])
    min_lag, max_lag = min(lags), max(lags)
    if min_lag == 0:
        return f'{base} through L{max_lag}'
    else:
        return f'L{min_lag} through L{max_lag} {base}'

def build_x_columns(var_defs):
    """Build list of X column names from variable definitions."""
    cols = []
    for base, lags in var_defs:
        for lag in lags:
            cols.append(make_col_name(base, lag))
    return cols

def get_group_indices(var_defs, has_constant=True):
    """Return dict mapping base variable names to their coefficient indices."""
    offset = 1 if has_constant else 0
    groups = {}
    idx = offset
    for base, lags in var_defs:
        groups[base] = list(range(idx, idx + len(lags)))
        idx += len(lags)
    return groups

def create_lags(df, var, max_lag=4):
    """Create lagged variables if they don't exist."""
    for i in range(1, max_lag + 1):
        col = f'L{i}_{var}'
        if col not in df.columns:
            df[col] = df[var].shift(i)

def constrained_regression(y, X, constraint_matrix, constraint_value, add_constant=True):
    """Perform constrained OLS regression."""
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
                self.rsquared = 1 - np.sum(residuals**2) / np.sum((y_clean - np.mean(y_clean))**2)
                # Calculate p-values from t-statistics
                t_stats = beta / se
                self.pvalues = 2 * (1 - sp_stats.t.cdf(np.abs(t_stats), n - k))

        results = ConstrainedResults()
    else:
        model = sm.OLS(y_clean, X_clean)
        results = model.fit()

    return results, data.index


def f_test_sum(results, param_indices, use_ols=False):
    """F-test for sum of coefficients = 0."""
    R = np.zeros((1, len(results.params)))
    R[0, param_indices] = 1
    if use_ols:
        f_result = results.f_test(R)
        return float(f_result.pvalue)
    else:
        q = np.array([0])
        f_stat = (R @ results.params - q).T @ np.linalg.inv(R @ results.cov_params_default @ R.T) @ (R @ results.params - q) / 1
        p_value = 1 - sp_stats.f.cdf(f_stat, 1, results.df_resid)
        return p_value[0] if isinstance(p_value, np.ndarray) else p_value


def f_test_joint(results, param_indices, use_ols=False):
    """F-test for joint significance."""
    k = len(param_indices)
    R = np.zeros((k, len(results.params)))
    for i, idx in enumerate(param_indices):
        R[i, idx] = 1
    if use_ols:
        f_result = results.f_test(R)
        return float(f_result.pvalue)
    else:
        q = np.zeros(k)
        f_stat = (R @ results.params - q).T @ np.linalg.inv(R @ results.cov_params_default @ R.T) @ (R @ results.params - q) / k
        p_value = 1 - sp_stats.f.cdf(f_stat, k, results.df_resid)
        return p_value


def compute_group_stats(results, var_defs, has_constant=True, use_ols=False):
    """Compute sum, p_sum, and p_joint for each variable group."""
    groups = get_group_indices(var_defs, has_constant)
    stats_dict = {}
    for base, indices in groups.items():
        coef_sum = sum(results.params[i] for i in indices)
        p_sum = f_test_sum(results, indices, use_ols)
        p_joint = f_test_joint(results, indices, use_ols)
        # Find lags for this group
        lags = [lags for b, lags in var_defs if b == base][0]
        label = make_group_label(base, lags)
        stats_dict[base] = {
            'label': label,
            'sum': coef_sum,
            'p_sum': p_sum,
            'p_joint': p_joint,
            'indices': indices,
        }
    return stats_dict


def make_lag_label(lags):
    """Generate a lag label like 'L1 to L4' or 'L1'."""
    if len(lags) == 1:
        return f'L{int(lags[0])}' if lags[0] > 0 else 'L0'
    min_lag, max_lag = int(min(lags)), int(max(lags))
    if min_lag == 0:
        return f'L0 to L{max_lag}'
    else:
        return f'L{min_lag} to L{max_lag}'


def build_summary_df(group_stats, var_defs, r2, nobs, r2_label='R2', extra_rows=None):
    """Build a transposed summary DataFrame with variables as columns."""
    # Get variable names in order
    var_names = [base for base, _ in var_defs]

    # Build the transposed data structure
    data = {'': []}  # First column for row labels

    for base in var_names:
        data[base] = []

    # Row 1: Lag
    data[''].append('Lag')
    for base, lags in var_defs:
        data[base].append(make_lag_label(lags))

    # Row 2: Sum of coefficients
    data[''].append('Sum of coefficients')
    for base in var_names:
        data[base].append(group_stats[base]['sum'])

    # Row 3: P-value (sum)
    data[''].append('P-value (sum)')
    for base in var_names:
        data[base].append(group_stats[base]['p_sum'])

    # Row 4: P-value (joint)
    data[''].append('P-value (joint)')
    for base in var_names:
        data[base].append(group_stats[base]['p_joint'])

    # Add extra rows if provided (e.g., long-run multipliers)
    if extra_rows:
        for row in extra_rows:
            data[''].append(row['label'])
            for i, base in enumerate(var_names):
                data[base].append(row['values'][i] if i < len(row['values']) else '')

    # Blank row
    data[''].append('')
    for base in var_names:
        data[base].append('')

    # R-squared row
    data[''].append(r2_label)
    data[var_names[0]].append(r2)
    for base in var_names[1:]:
        data[base].append('')

    # Number of observations row
    data[''].append('No. observations')
    data[var_names[0]].append(nobs)
    for base in var_names[1:]:
        data[base].append('')

    return pd.DataFrame(data)


def predict_out_of_sample(X_full, params, add_constant=True):
    """Generate predictions for full sample using estimated coefficients."""
    X_clean = X_full.values
    if add_constant:
        X_clean = sm.add_constant(X_clean)
    return X_clean @ params


#*******************************************************************************
# MAIN SCRIPT
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

# %%
# DATA MANIPULATION
print("Creating derived variables...")

# Convert Date
if 'Date' in df.columns:
    if not pd.api.types.is_datetime64_any_dtype(df['Date']):
        df['Date'] = pd.to_datetime(df['Date'].apply(lambda x: x.replace(" ", "") if isinstance(x, str) else x))
    df['period'] = df['Date']
    df = df.drop('Date', axis=1)

df = df.sort_values('period').reset_index(drop=True)

# Growth rates
df['gcpi'] = 400 * (np.log(df['CPIAUCSL']) - np.log(df['CPIAUCSL'].shift(1)))
df['gw'] = 400 * (np.log(df['ECIWAG']) - np.log(df['ECIWAG'].shift(1)))
df['gpty'] = 400 * (np.log(df['OPHNFB']) - np.log(df['OPHNFB'].shift(1)))
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
df['shortage'] = df['SHORTAGE'].fillna(5)
df['diffcpicf'] = 0.25 * sum([df['gcpi'].shift(i) for i in range(4)]) - df['cf1'].shift(4)

# COVID dummies
df['dummyq2_2020'] = 0.0
df['dummyq3_2020'] = 0.0
df['year'] = df['period'].dt.year
df['quarter'] = df['period'].dt.quarter
df.loc[(df['year'] == 2020) & (df['quarter'] == 2), 'dummyq2_2020'] = 1.0
df.loc[(df['year'] == 2020) & (df['quarter'] == 3), 'dummyq3_2020'] = 1.0

# New model variables
df['tcu'] = df['TCU']
df['ngdppot'] = df['NGDPPOT']
df['gscpi'] = df['GSCPI']

df['log_ngdppot'] = np.log(df['ngdppot'])
df['log_w'] = np.log(df['ECIWAG'])
df['log_tcu'] = np.log(df['tcu']/100)
df['log_tcu_trend'] = np.log(df['tcu'].rolling(window=40, min_periods=20).mean()/100)

if USE_LOG_CU_WAGES:
    df['cu'] = df['log_tcu'] - df['log_tcu_trend']
else:
    df['cu'] = (df['tcu'] - df['tcu'].rolling(window=40, min_periods=20).mean()) / 100

df['excess_demand'] = df['log_w'] - df['log_ngdppot'] - df['log_tcu']
df['excess_demand_og'] = df['excess_demand']
df['excess_demand_trend'] = df['excess_demand'].rolling(window=40).mean()
if USE_DETRENDED_EXCESS_DEMAND:
    df['excess_demand'] = df['excess_demand'] - df['excess_demand_trend']

# Filter to sample period
df = df[(df['period'] >= '1989-01-01') & (df['period'] <= '2023-06-30')].copy()
df = df.reset_index(drop=True)
pre_covid_mask = df['period'] <= '2019-12-31'

print(f"Sample size: {len(df)} observations")

# Create all needed lags
for var in ['gw', 'cf1', 'vu', 'diffcpicf', 'cu', 'shortage', 'excess_demand', 'gscpi', 'gcpi', 'grpe', 'grpf', 'cf10']:
    create_lags(df, var, max_lag=4)
df['L1_magpty'] = df['magpty'].shift(1)


# %% 
# plot some stuff
import seaborn as sns
import matplotlib.pyplot as plt
# set dpi to 300
plt.rcParams['figure.dpi'] = 300

# create 2x1 plot
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

sns.lineplot(ax=axes[0], x='period', y='excess_demand_og', data=df)
axes[0].set_xlabel('Date')
axes[0].set_ylabel('')  # keep blank if you want
axes[0].set_title('Excess Demand')

sns.lineplot(ax=axes[1], x='period', y='excess_demand', data=df)
axes[1].set_xlabel('Date')
axes[1].set_ylabel('')
axes[1].set_title('Excess Demand (Detrended)')

plt.tight_layout()
plt.savefig('excess_demand.pdf')
plt.show()
# %%
sns.lineplot(x='period', y='gscpi', data=df, label='Excess Demand')
plt.ylabel('')
plt.xlabel('Date')

# %%
sns.lineplot(x='period', y='excess_demand_og', data=df, label='Excess Demand')

# %%
fig, ax1 = plt.subplots()

ax1.set_xlabel('period')
ax1.set_ylabel('gw', color='tab:blue')
sns.lineplot(x='period', y='gw', data=df, ax=ax1, color='tab:blue')
ax1.tick_params(axis='y', labelcolor='tab:blue')

ax2 = ax1.twinx()
ax2.set_ylabel('cu', color='tab:orange')
sns.lineplot(x='period', y='cu', data=df, ax=ax2, color='tab:orange')
ax2.tick_params(axis='y', labelcolor='tab:orange')

plt.tight_layout()
plt.savefig('gw_cu.pdf')
plt.show()

# plt.title('Excess Demand')

# %%
#*******************************************************************************
# EQUATION 1: WAGE EQUATION
#*******************************************************************************
print("\n" + "="*80)
sample_label = "PRE-COVID SAMPLE" if USE_PRE_COVID_SAMPLE else "FULL SAMPLE"
print(f"EQUATION 1: WAGE EQUATION (gw) - {sample_label}")
print("="*80)

X_wage_cols = build_x_columns(WAGE_VARS)
if not USE_PRE_COVID_SAMPLE:
    X_wage_cols += ['dummyq2_2020', 'dummyq3_2020']

X_wage_full = df[X_wage_cols].copy()
y_wage_full = df['gw'].copy()

if USE_PRE_COVID_SAMPLE:
    X_wage = X_wage_full[pre_covid_mask].copy()
    y_wage = y_wage_full[pre_covid_mask].copy()
else:
    X_wage = X_wage_full.copy()
    y_wage = y_wage_full.copy()

# Constraint: sum of gw lags + sum of cf1 lags = 1
constraint_R = np.zeros((1, len(X_wage_cols)))
gw_indices = get_group_indices(WAGE_VARS, has_constant=False)['gw']
cf1_indices = get_group_indices(WAGE_VARS, has_constant=False)['cf1']
for i in gw_indices + cf1_indices:
    constraint_R[0, i] = 1
constraint_q = np.array([1.0])

print("Running constrained regression...")
results_wage, valid_idx = constrained_regression(y_wage, X_wage, constraint_R, constraint_q, add_constant=True)

if USE_PRE_COVID_SAMPLE:
    df['gwf1'] = predict_out_of_sample(X_wage_full, results_wage.params, add_constant=True)
else:
    df.loc[valid_idx, 'gwf1'] = results_wage.fittedvalues
df['gw_residuals'] = df['gw'] - df['gwf1']

# Compute statistics
wage_stats = compute_group_stats(results_wage, WAGE_VARS, has_constant=True)

# R-squared
valid_data = df.dropna(subset=['gw', 'gwf1'])
if USE_PRE_COVID_SAMPLE:
    valid_data = valid_data[(valid_data['period'] >= '1990-01-01') & (valid_data['period'] <= '2019-12-31')]
else:
    valid_data = valid_data[valid_data['period'] >= '1990-01-01']
r2_wage = np.corrcoef(valid_data['gw'], valid_data['gwf1'])[0, 1] ** 2

print(f"\nSum of coefficients:")
for base, stats in wage_stats.items():
    print(f"  {stats['label']}: {stats['sum']:.6f}")
print(f"\nR-squared: {r2_wage:.6f}")
print(f"Observations: {results_wage.nobs}")

# Save coefficients
coef_df = pd.DataFrame({'Variable': ['const'] + X_wage_cols, 'beta': results_wage.params, 'se': results_wage.bse, 'pvalue': results_wage.pvalues})
with pd.ExcelWriter(output_dir / f'eq_coefficients_new_model{output_suffix}.xlsx', engine='openpyxl', mode='w') as writer:
    coef_df.to_excel(writer, sheet_name='gw', index=False)

# %%
#*******************************************************************************
# EQUATION 2: SHORTAGE EQUATION
#*******************************************************************************
print("\n" + "="*80)
print("EQUATION 2: SHORTAGE EQUATION - FULL SAMPLE")
print("="*80)

X_shortage_cols = build_x_columns(SHORTAGE_VARS)
X_shortage = df[X_shortage_cols].copy()
y_shortage = df['shortage'].copy()

shortage_data = pd.concat([y_shortage, X_shortage], axis=1).dropna()
y_short_clean = shortage_data['shortage']
X_short_clean = shortage_data.drop('shortage', axis=1)

model_shortage = sm.OLS(y_short_clean, sm.add_constant(X_short_clean))
results_shortage = model_shortage.fit()

df.loc[shortage_data.index, 'shortagef'] = results_shortage.fittedvalues
df['shortage_residuals'] = df['shortage'] - df['shortagef']

shortage_stats = compute_group_stats(results_shortage, SHORTAGE_VARS, has_constant=True, use_ols=True)

# Long-run multipliers
sum_shortage_lag = shortage_stats['shortage']['sum']
sum_excess_demand = shortage_stats['excess_demand']['sum']
sum_gscpi = shortage_stats['gscpi']['sum']

if abs(1 - sum_shortage_lag) > 0.01:
    lr_excess_demand = sum_excess_demand / (1 - sum_shortage_lag)
    lr_gscpi = sum_gscpi / (1 - sum_shortage_lag)
else:
    lr_excess_demand = lr_gscpi = np.nan

print(f"\nSum of coefficients:")
for base, stats in shortage_stats.items():
    print(f"  {stats['label']}: {stats['sum']:.6f}")
print(f"\nLong-run multipliers:")
print(f"  Excess demand: {lr_excess_demand:.4f}")
print(f"  GSCPI: {lr_gscpi:.4f}")
print(f"\nR-squared: {results_shortage.rsquared:.6f}")
print(f"Observations: {results_shortage.nobs}")

coef_df = pd.DataFrame({'Variable': ['const'] + X_shortage_cols, 'beta': results_shortage.params, 'se': results_shortage.bse, 'pvalue': results_shortage.pvalues})
with pd.ExcelWriter(output_dir / f'eq_coefficients_new_model{output_suffix}.xlsx', engine='openpyxl', mode='a') as writer:
    coef_df.to_excel(writer, sheet_name='shortage', index=False)

# %%
#*******************************************************************************
# EQUATION 3: PRICE EQUATION
#*******************************************************************************
print("\n" + "="*80)
print("EQUATION 3: PRICE EQUATION (gcpi) - FULL SAMPLE")
print("="*80)

X_price_cols = build_x_columns(PRICE_VARS)
X_price = df[X_price_cols].copy()
y_price = df['gcpi'].copy()

# Constraint: sum of gcpi lags + sum of gw = 1
constraint_R = np.zeros((1, len(X_price_cols)))
price_groups = get_group_indices(PRICE_VARS, has_constant=False)
for i in price_groups['gcpi'] + price_groups['gw']:
    constraint_R[0, i] = 1
constraint_q = np.array([1.0])

print("Running constrained regression...")
results_price, valid_idx = constrained_regression(y_price, X_price, constraint_R, constraint_q, add_constant=True)

df.loc[valid_idx, 'gcpif'] = results_price.fittedvalues
df['gcpi_residuals'] = df['gcpi'] - df['gcpif']

price_stats = compute_group_stats(results_price, PRICE_VARS, has_constant=True)

valid_data = df.dropna(subset=['gcpi', 'gcpif'])
valid_data = valid_data[valid_data['period'] >= '1990-01-01']
r2_price = np.corrcoef(valid_data['gcpi'], valid_data['gcpif'])[0, 1] ** 2

print(f"\nSum of coefficients:")
for base, stats in price_stats.items():
    print(f"  {stats['label']}: {stats['sum']:.6f}")
print(f"\nR-squared: {r2_price:.6f}")
print(f"Observations: {results_price.nobs}")

coef_df = pd.DataFrame({'Variable': ['const'] + X_price_cols, 'beta': results_price.params, 'se': results_price.bse, 'pvalue': results_price.pvalues})
with pd.ExcelWriter(output_dir / f'eq_coefficients_new_model{output_suffix}.xlsx', engine='openpyxl', mode='a') as writer:
    coef_df.to_excel(writer, sheet_name='gcpi', index=False)

# %%
#*******************************************************************************
# EQUATION 4: CF1
#*******************************************************************************
print("\n" + "="*80)
sample_label = "PRE-COVID SAMPLE" if USE_PRE_COVID_SAMPLE else "FULL SAMPLE"
print(f"EQUATION 4: 1-YEAR EXPECTATIONS (cf1) - {sample_label}")
print("="*80)

X_cf1_cols = build_x_columns(CF1_VARS)
X_cf1_full = df[X_cf1_cols].copy()
y_cf1_full = df['cf1'].copy()

if USE_PRE_COVID_SAMPLE:
    X_cf1 = X_cf1_full[pre_covid_mask].copy()
    y_cf1 = y_cf1_full[pre_covid_mask].copy()
else:
    X_cf1 = X_cf1_full.copy()
    y_cf1 = y_cf1_full.copy()

constraint_R = np.ones((1, len(X_cf1_cols)))
constraint_q = np.array([1.0])

print("Running constrained regression (no constant)...")
results_cf1, valid_idx = constrained_regression(y_cf1, X_cf1, constraint_R, constraint_q, add_constant=False)

if USE_PRE_COVID_SAMPLE:
    df['cf1f'] = predict_out_of_sample(X_cf1_full, results_cf1.params, add_constant=False)
else:
    df.loc[valid_idx, 'cf1f'] = results_cf1.fittedvalues
df['cf1_residuals'] = df['cf1'] - df['cf1f']

cf1_stats = compute_group_stats(results_cf1, CF1_VARS, has_constant=False)

valid_data = df.dropna(subset=['cf1', 'cf1f'])
if USE_PRE_COVID_SAMPLE:
    valid_data = valid_data[(valid_data['period'] >= '1990-01-01') & (valid_data['period'] <= '2019-12-31')]
else:
    valid_data = valid_data[valid_data['period'] >= '1990-01-01']
r2_cf1 = np.corrcoef(valid_data['cf1'], valid_data['cf1f'])[0, 1] ** 2

print(f"\nSum of coefficients:")
for base, stats in cf1_stats.items():
    print(f"  {stats['label']}: {stats['sum']:.6f}")
print(f"\nR-squared: {r2_cf1:.6f}")
print(f"Observations: {results_cf1.nobs}")

coef_df = pd.DataFrame({'Variable': X_cf1_cols, 'beta': results_cf1.params, 'se': results_cf1.bse, 'pvalue': results_cf1.pvalues})
with pd.ExcelWriter(output_dir / f'eq_coefficients_new_model{output_suffix}.xlsx', engine='openpyxl', mode='a') as writer:
    coef_df.to_excel(writer, sheet_name='cf1', index=False)

# %%
#*******************************************************************************
# EQUATION 5: CF10
#*******************************************************************************
print("\n" + "="*80)
sample_label = "PRE-COVID SAMPLE" if USE_PRE_COVID_SAMPLE else "FULL SAMPLE"
print(f"EQUATION 5: 10-YEAR EXPECTATIONS (cf10) - {sample_label}")
print("="*80)

X_cf10_cols = build_x_columns(CF10_VARS)
X_cf10_full = df[X_cf10_cols].copy()
y_cf10_full = df['cf10'].copy()

if USE_PRE_COVID_SAMPLE:
    X_cf10 = X_cf10_full[pre_covid_mask].copy()
    y_cf10 = y_cf10_full[pre_covid_mask].copy()
else:
    X_cf10 = X_cf10_full.copy()
    y_cf10 = y_cf10_full.copy()

constraint_R = np.ones((1, len(X_cf10_cols)))
constraint_q = np.array([1.0])

print("Running constrained regression (no constant)...")
results_cf10, valid_idx = constrained_regression(y_cf10, X_cf10, constraint_R, constraint_q, add_constant=False)

if USE_PRE_COVID_SAMPLE:
    df['cf10f'] = predict_out_of_sample(X_cf10_full, results_cf10.params, add_constant=False)
else:
    df.loc[valid_idx, 'cf10f'] = results_cf10.fittedvalues
df['cf10_residuals'] = df['cf10'] - df['cf10f']

cf10_stats = compute_group_stats(results_cf10, CF10_VARS, has_constant=False)

valid_data = df.dropna(subset=['cf10', 'cf10f'])
if USE_PRE_COVID_SAMPLE:
    valid_data = valid_data[(valid_data['period'] >= '1990-01-01') & (valid_data['period'] <= '2019-12-31')]
else:
    valid_data = valid_data[valid_data['period'] >= '1990-01-01']
r2_cf10 = np.corrcoef(valid_data['cf10'], valid_data['cf10f'])[0, 1] ** 2

print(f"\nSum of coefficients:")
for base, stats in cf10_stats.items():
    print(f"  {stats['label']}: {stats['sum']:.6f}")
print(f"\nR-squared: {r2_cf10:.6f}")
print(f"Observations: {results_cf10.nobs}")

coef_df = pd.DataFrame({'Variable': X_cf10_cols, 'beta': results_cf10.params, 'se': results_cf10.bse, 'pvalue': results_cf10.pvalues})
with pd.ExcelWriter(output_dir / f'eq_coefficients_new_model{output_suffix}.xlsx', engine='openpyxl', mode='a') as writer:
    coef_df.to_excel(writer, sheet_name='cf10', index=False)

# %%
#*******************************************************************************
# EXPORT DATA
#*******************************************************************************
print("\n" + "="*80)
print("EXPORTING DATA")
print("="*80)

df['r2_wage'] = r2_wage
df['r2_shortage'] = results_shortage.rsquared
df['r2_price'] = r2_price
df['r2_cf1'] = r2_cf1
df['r2_cf10'] = r2_cf10

export_vars = ['period', 'gcpi', 'vu', 'gw', 'magpty', 'grpe', 'grpf', 'cf1', 'cf10',
               'shortage', 'diffcpicf', 'tcu', 'cu', 'ngdppot', 'gscpi', 'log_ngdppot',
               'log_w', 'excess_demand', 'excess_demand_trend',
               'gwf1', 'shortagef', 'gcpif', 'cf1f', 'cf10f',
               'gw_residuals', 'shortage_residuals', 'gcpi_residuals', 'cf1_residuals', 'cf10_residuals',
               'r2_wage', 'r2_shortage', 'r2_price', 'r2_cf1', 'r2_cf10']

if not USE_PRE_COVID_SAMPLE:
    export_vars.insert(11, 'dummyq2_2020')
    export_vars.insert(12, 'dummyq3_2020')

export_cols = [col for col in export_vars if col in df.columns]
df_export = df[export_cols].copy()
df_export.to_excel(output_dir / f'eq_simulations_data_new_model{output_suffix}.xlsx', index=False)
print(f"Exported data to {output_dir}")

# %%
#*******************************************************************************
# SUMMARY STATISTICS
#*******************************************************************************
print("\n" + "="*80)
print("CREATING SUMMARY STATISTICS")
print("="*80)

r2_label = 'R2 (pre-COVID)' if USE_PRE_COVID_SAMPLE else 'R2'

# Build summary DataFrames automatically from group stats (transposed format)
summary_wage = build_summary_df(wage_stats, WAGE_VARS, r2_wage, results_wage.nobs, r2_label)
summary_price = build_summary_df(price_stats, PRICE_VARS, r2_price, results_price.nobs, 'R2')
summary_cf1 = build_summary_df(cf1_stats, CF1_VARS, r2_cf1, results_cf1.nobs, r2_label)
summary_cf10 = build_summary_df(cf10_stats, CF10_VARS, r2_cf10, results_cf10.nobs, r2_label)

# Shortage summary with extra rows for long-run multipliers
shortage_extra = [
    {'label': 'Long-run multiplier', 'values': ['', lr_excess_demand, lr_gscpi]},
]
summary_shortage = build_summary_df(shortage_stats, SHORTAGE_VARS, results_shortage.rsquared, results_shortage.nobs, 'R2', extra_rows=shortage_extra)

summary_file = output_dir / f'summary_stats_new_model{output_suffix}.xlsx'
with pd.ExcelWriter(summary_file, engine='openpyxl', mode='w') as writer:
    summary_wage.to_excel(writer, sheet_name='gw', index=False)
    summary_shortage.to_excel(writer, sheet_name='shortage', index=False)
    summary_price.to_excel(writer, sheet_name='gcpi', index=False)
    summary_cf1.to_excel(writer, sheet_name='cf1', index=False)
    summary_cf10.to_excel(writer, sheet_name='cf10', index=False)

print(f"Saved summary statistics to {summary_file}")

# %%
# FINAL SUMMARY
print("\n" + "="*80)
print("MODEL ESTIMATION COMPLETE!")
print("="*80)

print(f"\n1. WAGE EQUATION: R² = {r2_wage:.4f}")
print(f"   CU effect (sum): {wage_stats['cu']['sum']:.4f}, p-value: {wage_stats['cu']['p_joint']:.4f}")

print(f"\n2. SHORTAGE EQUATION: R² = {results_shortage.rsquared:.4f}")
print(f"   Long-run ED multiplier: {lr_excess_demand:.4f}")
print(f"   Long-run GSCPI multiplier: {lr_gscpi:.4f}")

print(f"\n3. PRICE EQUATION: R² = {r2_price:.4f}")
print(f"\n4. CF1 EQUATION: R² = {r2_cf1:.4f}")
print(f"\n5. CF10 EQUATION: R² = {r2_cf10:.4f}")

print("\n" + "="*80)
print(f"Output saved to: {output_dir}")
print("="*80)
# %%
