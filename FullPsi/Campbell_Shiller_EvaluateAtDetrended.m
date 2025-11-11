clear; close all;

%% Load data
breakpoint = 1;
n_coint_var = 3;
LoadData_Benchmark;
ttime = [1947:2020];

output_rp = 0.03;
% Evaluate at detrended value
X2t = X2;

%% Estimation
fn = @(x)([x(1) * (1 - x(2)) - x(3) - (x0 + pi0 - y0nom_1 - yspr0 - output_rp), ...
               x(2) - exp(x(1)) / (exp(x(1)) + 1), ...
               x(3) - (log(exp(x(1)) + 1) - x(2) * x(1))]);
options = optimoptions('fsolve', 'TolX', 1e-4, 'MaxFunctionEvaluations', 1e5, 'MaxIterations', 1e5);
[x, fval] = fsolve(fn, [0, 0, 0], options);

pxbar = x(1);
k1x = exp(pxbar) / (exp(pxbar) + 1);
k0x = log(exp(pxbar) + 1) - k1x * pxbar;

pmbar = A0m;
k1m = exp(pmbar) / (exp(pmbar) + 1);
k0m = log(exp(pmbar) + 1) - k1m * pmbar;
erp0 = x0 + pi0 - y0nom_1 - yspr0 - (pmbar * (1 - k1m) - k0m);
erp = erp0 + (1 - k1m) * (- I_pdm' * X2t' + ((I_pi + I_gdp + I_divgrm)' * Psi - (I_y1 + I_yspr)') * inv(eye(N) - k1m * Psi) * X2t');

DR = (I_y1 + I_yspr)' * inv(eye(N) - k1x * Psi) * X2t';
DRM = (I_y1 + I_yspr)' * inv(eye(N) - k1m * Psi) * X2t' + (erp - erp0) / (1 - k1m);

CFX = (I_pi + I_gdp)' * Psi * inv(eye(N) - k1x * Psi) * X2t';
CFM = (I_pi + I_gdp + I_divgrm)' * Psi * inv(eye(N) - k1m * Psi) * X2t';
CFT = (I_pi + I_gdp + I_dt)' * Psi * inv(eye(N) - k1x * Psi) * X2t';
CFG = (I_pi + I_gdp + I_dg)' * Psi * inv(eye(N) - k1x * Psi) * X2t';

pdX = pxbar + CFX - DR;
pdT = pxbar + CFT - DR;
pdG = pxbar + CFG - DR;
pdM = pmbar + CFM - DRM;

s = exp(pdT) .* taxrevgdp' - exp(pdG) .* spendgdp';
upper = exp(pxbar) * (mean(taxrevgdp - spendgdp));

save('MAT/CS_EvaluateAtDetrended.mat');

s1 = s;

%% plot
f = figure;

load('MAT/CS30.mat');
plot(ttime, s, 'r', 'LineWidth', 3); hold on;

plot(ttime, s1, 'k--', 'LineWidth', 3); hold on;

load('MAT/CS30.mat');
plot(ttime, gdebt(2:end), 'b', 'LineWidth', 3); hold on;
axis([min(ttime) max(ttime) -3 3])

legend('Evaluated at Actual Values', 'Evaluated at Detrended Values', ...
    'Debt/GDP', 'Location', 'NorthWest')

xlabel('Year')
ylabel('PV(Surplus)/GDP')
set(gca, 'Layer', 'top', 'FontSize', 9)
set(gca, 'FontName', 'Times New Roman')

grid

f.PaperSize = [6 4];
print(['../figures/cs_EvaluateAtDetrended.pdf'], '-dpdf', '-fillpage');
