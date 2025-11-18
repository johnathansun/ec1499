%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Compute PV of Government Surplus Using Empirical Expectations
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% PURPOSE:
% This script computes the present value of government primary surplus using:
%   1. DISCOUNT RATES from the estimated asset pricing model (L0, L1)
%   2. CASH FLOW EXPECTATIONS from user-provided empirical data
%
% This allows you to test debt valuation using survey expectations, forecaster
% data, or any other empirical measure of expected future surpluses, while
% maintaining the model's sophisticated treatment of time-varying risk premia.
%
% INPUTS REQUIRED:
%   - Estimated pricing kernel from main_step3.m (automatic)
%   - User-provided surplus expectations (see Section 2 below)
%
% OUTPUTS:
%   - PV_surplus_empirical: Present value of surplus using empirical expectations
%   - valuation_gap: Difference between PV and government debt
%   - Plots comparing empirical PV to debt and model-based PV
%
% AUTHOR: Modified from original Jiang replication code
% DATE: 2025
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear; close all;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Section 1: Load Estimated Model and Data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fprintf('\n========================================\n');
fprintf('Loading estimated pricing kernel...\n');
fprintf('========================================\n\n');

% Load results from main_step3 (must be run first!)
if ~isfile('MAT/result_step3.mat')
    error(['ERROR: result_step3.mat not found!\n' ...
           'You must first run: main_step1, main_step2, main_step3\n' ...
           'to estimate the pricing kernel (L0, L1).']);
end

load('MAT/result_step3.mat');
LoadData_Benchmark;

fprintf('Successfully loaded:\n');
fprintf('  - Pricing kernel: L0 (size %dx1), L1 (size %dx%d)\n', length(L0), size(L1,1), size(L1,2));
fprintf('  - VAR parameters: Psi (size %dx%d)\n', size(Psi,1), size(Psi,2));
fprintf('  - Time periods: T = %d (years %d-%d)\n', T, min(date), max(date));
fprintf('  - Bond pricing coefficients: Api, Bpi (horizon = %d)\n', length(Api));

ttime = 1947:2020;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Section 2: LOAD YOUR EMPIRICAL SURPLUS EXPECTATIONS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% YOU MUST PROVIDE YOUR DATA HERE
%
% Your data should be a T x H matrix where:
%   - T = 74 rows (one for each year 1947-2020)
%   - H = forecast horizon in years (e.g., 50)
%   - surplus_expectations(t, j) = E_t[Surplus_{t+j}] / GDP_t
%
% The expectations should be in decimal form (e.g., 0.02 for 2% of GDP)
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fprintf('\n========================================\n');
fprintf('SECTION 2: Load Empirical Expectations\n');
fprintf('========================================\n\n');

%--------------------------------------------------------------------------
% OPTION A: Load from file (RECOMMENDED)
%--------------------------------------------------------------------------
% Uncomment and modify this section to load your data:

% fprintf('Loading empirical surplus expectations from file...\n');
% data_file = 'path/to/your/empirical_expectations.mat';
% empirical_data = load(data_file);
% surplus_expectations = empirical_data.surplus_expectations;  % Should be T x H

%--------------------------------------------------------------------------
% OPTION B: Load separate tax and spending expectations
%--------------------------------------------------------------------------
% If you have separate forecasts for taxes and spending:

% tax_expectations = empirical_data.tax_expectations;      % T x H matrix
% spending_expectations = empirical_data.spending_expectations;  % T x H matrix
% surplus_expectations = tax_expectations - spending_expectations;

%--------------------------------------------------------------------------
% OPTION C: Create dummy data for demonstration (DELETE THIS IN PRODUCTION)
%--------------------------------------------------------------------------

fprintf('WARNING: Using dummy data for demonstration!\n');
fprintf('Replace this with your actual empirical expectations.\n\n');

% Create synthetic expectations based on VAR but with some modifications
horizon = 50;  % Forecast horizon in years
surplus_expectations = zeros(T, horizon);

% Simple example: assume surplus/GDP mean-reverts to historical average
historical_avg_surplus = mean(surplusgdp);
persistence = 0.7;  % Speed of mean reversion

for t = 1:T
    current_surplus = surplusgdp(t);

    for j = 1:horizon
        % Mean-reverting expectation
        surplus_expectations(t, j) = historical_avg_surplus + ...
            persistence^j * (current_surplus - historical_avg_surplus);
    end
end

% END OF DUMMY DATA - REPLACE WITH YOUR ACTUAL DATA
%--------------------------------------------------------------------------

% Validate data dimensions
[nrows, ncols] = size(surplus_expectations);
fprintf('Empirical expectations loaded:\n');
fprintf('  - Dimensions: %d years x %d horizon\n', nrows, ncols);
fprintf('  - Sample values (year 1947, horizons 1-5): [');
fprintf('%.3f ', surplus_expectations(1, 1:5));
fprintf('...]\n');

if nrows ~= T
    error('ERROR: surplus_expectations must have %d rows (one per year 1947-2020)', T);
end

horizon = ncols;
fprintf('  - Using forecast horizon H = %d years\n', horizon);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Section 3: Compute Model-Implied Discount Factors
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fprintf('\n========================================\n');
fprintf('Computing discount factors...\n');
fprintf('========================================\n\n');

% Discount factors are the model-implied prices of zero-coupon bonds
% discount_factors(t, j) = price at time t of $1 received in j years
%                        = exp(A_bond(j) + B_bond(:,j)' * X_t)

discount_factors = zeros(T, horizon);
discount_factors_real = zeros(T, horizon);  % Real discount factors (TIPS)

for j = 1:min(horizon, length(Api))
    % Nominal discount factors (using Treasury pricing)
    discount_factors(:, j) = exp(Api(j) + Bpi(:, j)' * X2t')';

    % Real discount factors (using TIPS pricing)
    if j <= length(A)
        discount_factors_real(:, j) = exp(A(j) + B(:, j)' * X2t')';
    end
end

% For horizons beyond the computed bond coefficients, extrapolate
if horizon > length(Api)
    fprintf('WARNING: Horizon (%d) exceeds computed bond maturities (%d)\n', horizon, length(Api));
    fprintf('         Extrapolating discount factors for longer maturities...\n');

    % Simple extrapolation: assume constant long-term discount rate
    long_rate = -Api(end) / length(Api);  % Average yield on longest bond

    for j = (length(Api)+1):horizon
        discount_factors(:, j) = discount_factors(:, length(Api)) .* exp(-long_rate * (j - length(Api)));
        if j <= length(A)
            discount_factors_real(:, j) = discount_factors_real(:, length(A)) .* exp(-long_rate * (j - length(A)));
        end
    end
end

fprintf('Discount factors computed:\n');
fprintf('  - 1-year discount factor (avg): %.4f\n', mean(discount_factors(:, 1)));
fprintf('  - 10-year discount factor (avg): %.4f\n', mean(discount_factors(:, 10)));
fprintf('  - %d-year discount factor (avg): %.4f\n', horizon, mean(discount_factors(:, horizon)));

% Implied average discount rates (for reference)
avg_1yr_rate = -log(mean(discount_factors(:, 1))) * 100;
avg_10yr_rate = -log(mean(discount_factors(:, 10))) / 10 * 100;
fprintf('  - Implied 1-year rate (avg): %.2f%%\n', avg_1yr_rate);
fprintf('  - Implied 10-year rate (avg): %.2f%%\n', avg_10yr_rate);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Section 4: Compute Present Value Using Empirical Expectations
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fprintf('\n========================================\n');
fprintf('Computing PV(Surplus) with empirical expectations...\n');
fprintf('========================================\n\n');

% PV(Surplus)_t = Σ_{j=1}^H E_t^empirical[Surplus_{t+j}] × DiscountFactor_t(j)

PV_surplus_empirical = zeros(T, 1);
PV_surplus_components = zeros(T, horizon);  % For diagnostics

for t = 1:T
    pv_t = 0;

    for j = 1:horizon
        % Expected surplus j years ahead (from your empirical data)
        expected_surplus_j = surplus_expectations(t, j);

        % Model-implied discount factor
        discount_j = discount_factors(t, j);

        % Present value contribution from horizon j
        pv_contribution = expected_surplus_j * discount_j;
        pv_t = pv_t + pv_contribution;

        PV_surplus_components(t, j) = pv_contribution;
    end

    PV_surplus_empirical(t) = pv_t;
end

fprintf('PV(Surplus) computed:\n');
fprintf('  - Mean PV/GDP: %.3f (%.1f%%)\n', mean(PV_surplus_empirical), mean(PV_surplus_empirical)*100);
fprintf('  - Min PV/GDP: %.3f (year %d)\n', min(PV_surplus_empirical), ttime(PV_surplus_empirical == min(PV_surplus_empirical)));
fprintf('  - Max PV/GDP: %.3f (year %d)\n', max(PV_surplus_empirical), ttime(PV_surplus_empirical == max(PV_surplus_empirical)));
fprintf('  - Std Dev: %.3f\n', std(PV_surplus_empirical));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Section 5: Compare to VAR-Based Model Estimates
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fprintf('\n========================================\n');
fprintf('Comparing to VAR-based model...\n');
fprintf('========================================\n\n');

% Compute VAR-based PV (from Model_Summary)
PV_surplus_VAR = (PDt_model) .* taxrevgdp - (PDg_model) .* spendgdp;

fprintf('Comparison of methods:\n');
fprintf('  - Empirical expectations (mean): %.3f\n', mean(PV_surplus_empirical));
fprintf('  - VAR-based model (mean): %.3f\n', mean(PV_surplus_VAR));
fprintf('  - Difference (mean): %.3f\n', mean(PV_surplus_empirical - PV_surplus_VAR));
fprintf('  - Correlation: %.3f\n', corr(PV_surplus_empirical, PV_surplus_VAR));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Section 6: Test Debt Valuation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fprintf('\n========================================\n');
fprintf('Testing debt over/under-valuation...\n');
fprintf('========================================\n\n');

% Valuation gap = PV(Surplus) - Debt
% Positive gap → Debt is undervalued (surplus backs more than debt value)
% Negative gap → Debt is overvalued (surplus insufficient to back debt)

valuation_gap = PV_surplus_empirical - gdebt(2:end)';

fprintf('Debt valuation results:\n');
fprintf('  - Mean debt/GDP: %.3f\n', mean(gdebt(2:end)));
fprintf('  - Mean PV(Surplus)/GDP: %.3f\n', mean(PV_surplus_empirical));
fprintf('  - Mean valuation gap: %.3f\n', mean(valuation_gap));
fprintf('\n');
fprintf('  - Valuation gap in 1950s (avg): %.3f\n', mean(valuation_gap(1:10)));
fprintf('  - Valuation gap in 2010s (avg): %.3f\n', mean(valuation_gap(end-9:end)));
fprintf('\n');

% Count years with over/undervaluation
years_undervalued = sum(valuation_gap > 0);
years_overvalued = sum(valuation_gap < 0);

fprintf('  - Years with debt undervalued: %d (%.1f%%)\n', years_undervalued, years_undervalued/T*100);
fprintf('  - Years with debt overvalued: %d (%.1f%%)\n', years_overvalued, years_overvalued/T*100);

% Recent trend
if valuation_gap(end) < -0.5
    fprintf('\n  >> WARNING: Large negative gap in 2020 (%.3f)\n', valuation_gap(end));
    fprintf('     Debt appears significantly OVERVALUED relative to expected surpluses.\n');
elseif valuation_gap(end) > 0.5
    fprintf('\n  >> Large positive gap in 2020 (%.3f)\n', valuation_gap(end));
    fprintf('     Debt appears significantly UNDERVALUED relative to expected surpluses.\n');
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Section 7: Visualization
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fprintf('\n========================================\n');
fprintf('Creating plots...\n');
fprintf('========================================\n\n');

%--------------------------------------------------------------------------
% Figure 1: PV(Surplus) vs Debt Over Time
%--------------------------------------------------------------------------
f1 = figure('Position', [100, 100, 800, 500]);

plot(ttime, PV_surplus_empirical, 'b-', 'LineWidth', 3); hold on;
plot(ttime, PV_surplus_VAR, 'g--', 'LineWidth', 2);
plot(ttime, gdebt(2:end)', 'k:', 'LineWidth', 3);
plot(ttime, mean(PV_surplus_empirical) * ones(size(ttime)), 'r--', 'LineWidth', 1);

xlim([min(ttime), max(ttime)]);
xlabel('Year', 'FontSize', 12);
ylabel('Ratio to GDP', 'FontSize', 12);
title('Present Value of Government Surplus vs Debt', 'FontSize', 14);
legend('PV(Surplus) - Empirical Expectations', ...
       'PV(Surplus) - VAR Model', ...
       'Government Debt/GDP', ...
       'Mean PV (Empirical)', ...
       'Location', 'best');
grid on;
set(gca, 'FontName', 'Times New Roman', 'FontSize', 11);

f1.PaperSize = [8, 5];
print('../figures/pv_empirical_vs_debt', '-dpdf', '-fillpage');
fprintf('  - Saved: figures/pv_empirical_vs_debt.pdf\n');

%--------------------------------------------------------------------------
% Figure 2: Valuation Gap Over Time
%--------------------------------------------------------------------------
f2 = figure('Position', [150, 150, 800, 500]);

plot(ttime, valuation_gap, 'r-', 'LineWidth', 3); hold on;
yline(0, 'k--', 'LineWidth', 2);
fill([ttime, fliplr(ttime)], [zeros(size(ttime)), fliplr(valuation_gap')], 'r', ...
     'FaceAlpha', 0.2, 'EdgeColor', 'none');

xlim([min(ttime), max(ttime)]);
xlabel('Year', 'FontSize', 12);
ylabel('PV(Surplus) - Debt (ratio to GDP)', 'FontSize', 12);
title('Government Debt Valuation Gap', 'FontSize', 14);
grid on;
set(gca, 'FontName', 'Times New Roman', 'FontSize', 11);

% Add text annotation
text(0.05, 0.95, 'Positive = Debt Undervalued', ...
     'Units', 'normalized', 'FontSize', 10, 'VerticalAlignment', 'top');
text(0.05, 0.05, 'Negative = Debt Overvalued', ...
     'Units', 'normalized', 'FontSize', 10, 'VerticalAlignment', 'bottom');

f2.PaperSize = [8, 5];
print('../figures/valuation_gap_empirical', '-dpdf', '-fillpage');
fprintf('  - Saved: figures/valuation_gap_empirical.pdf\n');

%--------------------------------------------------------------------------
% Figure 3: Contribution by Horizon
%--------------------------------------------------------------------------
f3 = figure('Position', [200, 200, 800, 500]);

% Show how different horizons contribute to PV for a few select years
years_to_show = [1960, 1980, 2000, 2020];
colors = {'b', 'g', 'r', 'm'};

for i = 1:length(years_to_show)
    year_idx = find(ttime == years_to_show(i));
    if ~isempty(year_idx)
        plot(1:horizon, cumsum(PV_surplus_components(year_idx, :)), ...
             'Color', colors{i}, 'LineWidth', 2, 'DisplayName', sprintf('%d', years_to_show(i)));
        hold on;
    end
end

xlabel('Forecast Horizon (years)', 'FontSize', 12);
ylabel('Cumulative PV Contribution', 'FontSize', 12);
title('Build-up of PV(Surplus) by Forecast Horizon', 'FontSize', 14);
legend('Location', 'southeast');
grid on;
set(gca, 'FontName', 'Times New Roman', 'FontSize', 11);

f3.PaperSize = [8, 5];
print('../figures/pv_horizon_contribution', '-dpdf', '-fillpage');
fprintf('  - Saved: figures/pv_horizon_contribution.pdf\n');

%--------------------------------------------------------------------------
% Figure 4: Comparison - Empirical vs VAR Expectations
%--------------------------------------------------------------------------
f4 = figure('Position', [250, 250, 800, 500]);

scatter(PV_surplus_VAR, PV_surplus_empirical, 50, ttime, 'filled');
hold on;
plot([min(PV_surplus_VAR), max(PV_surplus_VAR)], ...
     [min(PV_surplus_VAR), max(PV_surplus_VAR)], 'k--', 'LineWidth', 2);

xlabel('PV(Surplus) - VAR Model', 'FontSize', 12);
ylabel('PV(Surplus) - Empirical Expectations', 'FontSize', 12);
title('Comparison of PV Estimates', 'FontSize', 14);
colorbar('Ticks', [1950, 1970, 1990, 2010], 'TickLabels', {'1950', '1970', '1990', '2010'});
grid on;
set(gca, 'FontName', 'Times New Roman', 'FontSize', 11);

f4.PaperSize = [8, 5];
print('../figures/pv_empirical_vs_var_scatter', '-dpdf', '-fillpage');
fprintf('  - Saved: figures/pv_empirical_vs_var_scatter.pdf\n');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Section 8: Save Results
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fprintf('\n========================================\n');
fprintf('Saving results...\n');
fprintf('========================================\n\n');

save('MAT/result_empirical_expectations.mat', ...
     'PV_surplus_empirical', 'valuation_gap', ...
     'surplus_expectations', 'discount_factors', ...
     'PV_surplus_components', 'PV_surplus_VAR', ...
     'ttime', 'horizon');

fprintf('Results saved to: MAT/result_empirical_expectations.mat\n');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Section 9: Summary Statistics Table
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fprintf('\n========================================\n');
fprintf('SUMMARY STATISTICS\n');
fprintf('========================================\n\n');

fprintf('%-40s %10s %10s %10s\n', 'Variable', 'Mean', 'Std Dev', '2020');
fprintf('%s\n', repmat('-', 1, 70));
fprintf('%-40s %10.3f %10.3f %10.3f\n', 'PV(Surplus) - Empirical', mean(PV_surplus_empirical), std(PV_surplus_empirical), PV_surplus_empirical(end));
fprintf('%-40s %10.3f %10.3f %10.3f\n', 'PV(Surplus) - VAR Model', mean(PV_surplus_VAR), std(PV_surplus_VAR), PV_surplus_VAR(end));
fprintf('%-40s %10.3f %10.3f %10.3f\n', 'Government Debt/GDP', mean(gdebt(2:end)), std(gdebt(2:end)), gdebt(end));
fprintf('%-40s %10.3f %10.3f %10.3f\n', 'Valuation Gap (PV - Debt)', mean(valuation_gap), std(valuation_gap), valuation_gap(end));
fprintf('%s\n', repmat('-', 1, 70));

fprintf('\n========================================\n');
fprintf('ANALYSIS COMPLETE\n');
fprintf('========================================\n\n');

fprintf('Next steps:\n');
fprintf('  1. Replace dummy data in Section 2 with your actual empirical expectations\n');
fprintf('  2. Review plots in ../figures/ directory\n');
fprintf('  3. Examine saved results in MAT/result_empirical_expectations.mat\n');
fprintf('  4. Adjust horizon or risk parameters if needed\n\n');
