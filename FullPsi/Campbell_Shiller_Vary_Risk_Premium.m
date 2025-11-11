clear; close all; 

%% Load data
LoadData_Benchmark;
ttime = [1947:2020];
rps = 0.015:0.001:0.06;

idx = 1;

for output_rp = rps
    %% Estimation
    fn = @(x)([x(1) * (1 - x(2)) - x(3) - (x0 + pi0 - y0nom_1 - yspr0 - output_rp), ...
                   x(2) - exp(x(1)) / (exp(x(1)) + 1), ...
                   x(3) - (log(exp(x(1)) + 1) - x(2) * x(1))]);
    options = optimoptions('fsolve', 'TolX', 1e-4, 'MaxFunctionEvaluations', 1e5, 'MaxIterations', 1e5);
    [x, fval] = fsolve(fn, [0, 0, 0], options);

    pxbar = x(1);
    k1x = exp(pxbar) / (exp(pxbar) + 1);
    k0x = log(exp(pxbar) + 1) - k1x * pxbar;

    fn = @(x)([x(1) * (1 - x(2)) - x(3) - (x0 + pi0 - y0nom_1 - yspr0 - output_rp / lev), ...
                   x(2) - exp(x(1)) / (exp(x(1)) + 1), ...
                   x(3) - (log(exp(x(1)) + 1) - x(2) * x(1))]);
    options = optimoptions('fsolve', 'TolX', 1e-4, 'MaxFunctionEvaluations', 1e5, 'MaxIterations', 1e5);
    [x, fval] = fsolve(fn, [0, 0, 0], options);

    pmbar = x(1);
    k1m = exp(pmbar) / (exp(pmbar) + 1);
    k0m = log(exp(pmbar) + 1) - k1m * pmbar;

    DR = (I_y1 + I_yspr)' * inv(eye(N) - k1x * Psi) * X2t';
    DRM = (I_y1 + I_yspr)' * inv(eye(N) - k1m * Psi) * X2t';

    CFX = (I_pi + I_gdp)' * Psi * inv(eye(N) - k1x * Psi) * X2t';
    CFM = (I_pi + I_gdp + I_divgrm)' * Psi * inv(eye(N) - k1m * Psi) * X2t';
    CFT = (I_pi + I_gdp + I_dt)' * Psi * inv(eye(N) - k1x * Psi) * X2t';
    CFG = (I_pi + I_gdp + I_dg)' * Psi * inv(eye(N) - k1x * Psi) * X2t';

    pdX(idx, :) = pxbar + CFX - DR;
    pdT = pxbar + CFT - DR;
    pdG = pxbar + CFG - DR;
    pdM(idx, :) = pmbar + CFM - DRM;

    s = exp(pdT) .* taxrevgdp' - exp(pdG) .* spendgdp';
    upper = exp(pxbar) * (mean(taxrevgdp - spendgdp));

    sall(idx, :) = s;
    gapmean(idx) = mean(gdebt(2:end) - s);
    smean(idx) = mean(s);
    idx = idx + 1;
end

s1 = s;

%% plot
f = figure;

plot(rps * 100, smean, 'b', 'LineWidth', 3);
xlabel('Risk Premium (%)')
ylabel('Mean PV(Surplus)/GDP Upper Bound')
set(gca, 'Layer', 'top', 'FontSize', 9)
set(gca, 'FontName', 'Times New Roman')
grid

f.PaperSize = [6 4];
print(['../figures/cs_vary_rp.pdf'], '-dpdf', '-fillpage');

f = figure;

plot(ttime, exp(pdX(1, :)), ttime, exp(pdM(1, :)), 'r', 'LineWidth', 3);
xlabel('Time')
ylabel('P/D Ratio')
legend('GDP claim', 'Equity')
set(gca, 'Layer', 'top', 'FontSize', 9)
set(gca, 'FontName', 'Times New Roman')
xlim([1947, 2020])
grid

f.PaperSize = [6 4];
print(['../figures/cs_rp015_pdx.pdf'], '-dpdf', '-fillpage');
