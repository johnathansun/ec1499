%% This code plots Cash Flow Forecasts
clear; close all;

LoadData_Benchmark;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 1-year
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

c = [c(1:5) c(divgrpos) c(pdpos) c(dtpos) c(dtpos) c(dgpos) c(dgpos)];
pred_coint = c' + Psi * X2t(1:(T - 1), :)';

regx_detrend = [ones(T - 1, 1), X2(1:(T - 1), :)];
regx_actual = [ones(T - 1, 1), X2t(1:(T - 1), :)];
regr = ols(X2(2:T, dtpos), regx_detrend);
fitted_dt = regx_actual * regr.beta;
regr = ols(X2(2:T, dgpos), regx_detrend);
fitted_dg = regx_actual * regr.beta;

ts_actual_dt = X2t(2:T, dtpos);
ts_var1y_dt = pred_coint(dtpos, :)';
ts_ols1y_dt = fitted_dt;

rmse1y_dt = [sqrt(mean((ts_actual_dt - ts_var1y_dt) .^ 2)), ...
                 sqrt(mean((ts_actual_dt - ts_ols1y_dt) .^ 2))] * 1e2;

ts_actual_dg = X2t(2:T, dgpos);
ts_var1y_dg = pred_coint(dgpos, :)';
ts_ols1y_dg = fitted_dg;

rmse1y_dg = [sqrt(mean((ts_actual_dg - ts_var1y_dg) .^ 2)), ...
                 sqrt(mean((ts_actual_dg - ts_ols1y_dg) .^ 2))] * 1e2;

ttime = [1947:2020];

f = figure;
plot(ttime(2:T), ts_actual_dt, 'b', ttime(2:T), ts_var1y_dt, 'r--', ttime(2:T), ts_ols1y_dt, 'k-.', 'LineWidth', 3)
legend('Data', ['VAR, rmse=', num2str(rmse1y_dt(1))], ...
    ['OLS, rmse=', num2str(rmse1y_dt(2))], 'location', 'northwest')
set(gca, 'Layer', 'top', 'FontSize', 9)
set(gca, 'FontName', 'Times New Roman')
grid
ylabel('\Delta Log Tax/GDP')
xlim([1947, 2020])
f.PaperSize = [6 3.75];
print('../figures/pic-forecast_1yr_dt', '-dpdf', '-fillpage');

f = figure;
plot(ttime(2:T), ts_actual_dg, 'b', ttime(2:T), ts_var1y_dg, 'r--', ttime(2:T), ts_ols1y_dg, 'k-.', 'LineWidth', 3)
legend('Data', ['VAR, rmse=', num2str(rmse1y_dg(1))], ...
    ['OLS, rmse=', num2str(rmse1y_dg(2))], 'location', 'northwest')
set(gca, 'Layer', 'top', 'FontSize', 9)
set(gca, 'FontName', 'Times New Roman')
grid
ylabel('\Delta Log Spending/GDP')
xlim([1947, 2020])
f.PaperSize = [6 3.75];
print('../figures/pic-forecast_1yr_dg', '-dpdf', '-fillpage');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 5-year
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Psi5c = eye(N);
Psi5 = Psi;

for i = 2:5
    Psi5c = Psi5c + Psi ^ (i - 1);
    Psi5 = Psi5 + Psi ^ i;
end

pred_coint = Psi5c * c' + Psi5 * X2t(1:(T - 5), :)';

X5 = X2(2:T - 4, :) + X2(3:T - 3, :) + X2(4:T - 2, :) + X2(5:T - 1, :) + X2(6:T, :);
X5t = X2t(2:T - 4, :) + X2t(3:T - 3, :) + X2t(4:T - 2, :) + X2t(5:T - 1, :) + X2t(6:T, :);

regx_detrend = [ones(T - 5, 1), X2(1:(T - 5), :)];
regx_actual = [ones(T - 5, 1), X2t(1:(T - 5), :)];
regr = ols(X5(:, dtpos), regx_detrend);
fitted_dt = regx_actual * regr.beta;
regr = ols(X5(:, dgpos), regx_detrend);
fitted_dg = regx_actual * regr.beta;

ts_actual_dt = X5t(:, dtpos);
ts_var5y_dt = pred_coint(dtpos, :)';
ts_ols5y_dt = fitted_dt;

rmse5y_dt = [sqrt(mean((ts_actual_dt - ts_var5y_dt) .^ 2)), ...
                 sqrt(mean((ts_actual_dt - ts_ols5y_dt) .^ 2))] * 1e2;

ts_actual_dg = X5t(:, dgpos);
ts_var5y_dg = pred_coint(dgpos, :)';
ts_ols5y_dg = fitted_dg;

rmse5y_dg = [sqrt(mean((ts_actual_dg - ts_var5y_dg) .^ 2)), ...
                 sqrt(mean((ts_actual_dg - ts_ols5y_dg) .^ 2))] * 1e2;

f = figure;
plot(ttime(6:T), ts_actual_dt, 'b', ttime(6:T), ts_var5y_dt, 'r--', ttime(6:T), ts_ols5y_dt, 'k-.', 'LineWidth', 3)
legend('Data', ['VAR, rmse=', num2str(rmse5y_dt(1))], ...
    ['OLS, rmse=', num2str(rmse5y_dt(2))], 'location', 'northwest')
set(gca, 'Layer', 'top', 'FontSize', 9)
set(gca, 'FontName', 'Times New Roman')
grid
ylabel('\Delta Log Tax/GDP')
xlim([1947, 2020])
f.PaperSize = [6 3.75];
print('../figures/pic-forecast_5yr_dt', '-dpdf', '-fillpage');

f = figure;
plot(ttime(6:T), ts_actual_dg, 'b', ttime(6:T), ts_var5y_dg, 'r--', ttime(6:T), ts_ols5y_dg, 'k-.', 'LineWidth', 3)
legend('Data', ['VAR, rmse=', num2str(rmse5y_dg(1))], ...
    ['OLS, rmse=', num2str(rmse5y_dg(2))], 'location', 'northwest')
set(gca, 'Layer', 'top', 'FontSize', 9)
set(gca, 'FontName', 'Times New Roman')
grid
ylabel('\Delta Log Spending/GDP')
xlim([1947, 2020])
f.PaperSize = [6 3.75];
print('../figures/pic-forecast_5yr_dg', '-dpdf', '-fillpage');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 10-year
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Psi10c = eye(N);
Psi10 = Psi;

for i = 2:10
    Psi10c = Psi10c + Psi ^ (i - 1);
    Psi10 = Psi10 + Psi ^ i;
end

pred_coint = Psi10c * c' + Psi10 * X2t(1:(T - 10), :)';

X10t = X2t(2:(T - 9), :);
X10 = X2(2:(T - 9), :);

for i = 2:10
    X10t = X10t + X2t((i + 1):(T - 10 + i), :);
    X10 = X10 + X2((i + 1):(T - 10 + i), :);
end

regx_detrend = [ones(T - 10, 1), X2(1:(T - 10), :)];
regx_actual = [ones(T - 10, 1), X2t(1:(T - 10), :)];
regr = ols(X10(:, dtpos), regx_detrend);
fitted_dt = regx_actual * regr.beta;
regr = ols(X10(:, dgpos), regx_detrend);
fitted_dg = regx_actual * regr.beta;

ts_actual_dt = X10t(:, dtpos);
ts_var10y_dt = pred_coint(dtpos, :)';
ts_ols10y_dt = fitted_dt;

rmse10y_dt = [sqrt(mean((ts_actual_dt - ts_var10y_dt) .^ 2)), ...
                  sqrt(mean((ts_actual_dt - ts_ols10y_dt) .^ 2))] * 1e2;

ts_actual_dg = X10t(:, dgpos);
ts_var10y_dg = pred_coint(dgpos, :)';
ts_ols10y_dg = fitted_dg;

rmse10y_dg = [sqrt(mean((ts_actual_dg - ts_var10y_dg) .^ 2)), ...
                  sqrt(mean((ts_actual_dg - ts_ols10y_dg) .^ 2))] * 1e2;

f = figure;
plot(ttime(11:T), ts_actual_dt, 'b', ttime(11:T), ts_var10y_dt, 'r--', ttime(11:T), ts_ols10y_dt, 'k-.', 'LineWidth', 3)
legend('Data', ['VAR, rmse=', num2str(rmse10y_dt(1))], ...
    ['OLS, rmse=', num2str(rmse10y_dt(2))], 'location', 'northwest')
set(gca, 'Layer', 'top', 'FontSize', 9)
set(gca, 'FontName', 'Times New Roman')
grid
ylabel('\Delta Log Tax/GDP')
xlim([1947, 2020])
f.PaperSize = [6 3.75];
print('../figures/pic-forecast_10yr_dt', '-dpdf', '-fillpage');

f = figure;
plot(ttime(11:T), ts_actual_dg, 'b', ttime(11:T), ts_var10y_dg, 'r--', ttime(11:T), ts_ols10y_dg, 'k-.', 'LineWidth', 3)
legend('Data', ['VAR, rmse=', num2str(rmse10y_dg(1))], ...
    ['OLS, rmse=', num2str(rmse10y_dg(2))], 'location', 'northwest')
set(gca, 'Layer', 'top', 'FontSize', 9)
set(gca, 'FontName', 'Times New Roman')
grid
ylabel('\Delta Log Spending/GDP')
xlim([1947, 2020])
f.PaperSize = [6 3.75];
print('../figures/pic-forecast_10yr_dg', '-dpdf', '-fillpage');
