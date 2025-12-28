# Repository Guidelines

## Project Structure & Module Organization
- Root contains the replication package and supporting docs. Primary work lives in `Replication Package/`.
- Data and analysis follow a numbered workflow under `Replication Package/Code and Data/`:
  - `(1) Data/` for public/private inputs.
  - `(2) Regressions/` for Stata/Python regression outputs and “Predicted vs. Actual” plots.
  - `(3) Core Results/` for MATLAB simulations and R/Python plotting.
  - `(4) Other Results/` for PCA and the Dynare simple model.
  - `(5) Appendix/` for robustness/alternative specifications.
- Figures and tables are in `Replication Package/Figures/` and `Replication Package/Tables/`.

## Build, Test, and Development Commands
This is a research replication repo; there is no single build target. Run steps in order and update paths in scripts.
- Stata regressions:
  - `cd "Replication Package/Code and Data/(2) Regressions"` then `do regression_full.do`
- Predicted vs actual plots (R):
  - `cd "Replication Package/Code and Data/(2) Regressions/Predicted vs. Actual"`
  - `Rscript plot_pred_v_actual.R`
- MATLAB core results (examples):
  - `cd "Replication Package/Code and Data/(3) Core Results/IRFs"` then run `irf_simulations.m`
- Python/R plots for PCA:
  - `cd "Replication Package/Code and Data/(4) Other Results/PCA"` then `python pca.py` and `Rscript plot_pca.R`

## Coding Style & Naming Conventions
- Match the existing style in each language.
- Python uses 4-space indentation and `snake_case` (e.g., `cond_forecast.py`).
- R scripts typically use 2-space indentation and `plot_*.R` naming.
- Stata/MATLAB scripts include `CHANGE PATH HERE` blocks; keep those at the top and update paths explicitly.
- Output naming follows `eq_coefficients_*.xlsx`, `summary_stats_*.xlsx`, and `results_*.xlsx` patterns.

## Testing Guidelines
- There are no automated unit tests. Validation is via replication outputs.
- Re-run the affected pipeline step and compare outputs to expected figures in `Replication Package/Figures/` or `Replication Package/Code and Data/*/Figures*`.
- When adjusting code that writes Excel/CSV outputs, confirm schema consistency (sheet names, column order).

## Commit & Pull Request Guidelines
- Recent commits use short, imperative summaries (e.g., “Refactor regressions”). Keep the first line concise.
- PRs should include a brief summary, list of touched pipeline steps, and any changed outputs (figures/tables) when applicable.
- Note required external data (Bloomberg/Haver) and avoid committing private data files from `Replication Package/Code and Data/(1) Data/Private Data/`.

## Data Access & Configuration Notes
- Scripts often require manual path updates (`setwd()`, `os.chdir()`, or `cd` in MATLAB/Stata). Update these consistently.
- Private data is required for PCA; document any substituted inputs when reproducing results.
