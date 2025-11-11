clear; close all;

LoadData_Benchmark;
output_rp = 0.03;

% Estimation
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

t = 2;
EpdT = pxbar + (I_pi + I_gdp + I_dt)' * Psi * inv(eye(N) - k1x * Psi) * (Psi * X2t(t - 1:end - t + 1, :)') ...
    - (I_y1 + I_yspr)' * inv(eye(N) - k1x * Psi) * (Psi * X2t(t - 1:end - t + 1, :)');
EpdG = pxbar + (I_pi + I_gdp + I_dg)' * Psi * inv(eye(N) - k1x * Psi) * (Psi * X2t(t - 1:end - t + 1, :)') ...
    - (I_y1 + I_yspr)' * inv(eye(N) - k1x * Psi) * (Psi * X2t(t - 1:end - t + 1, :)');
Etau = I_coint_tax' * Psi * X2t(t - 1:end - t + 1, :)';
Eg = I_coint_spending' * Psi * X2t(t - 1:end - t + 1, :)';
Vartau = (I_coint_tax)' * Sig * Sig' * (I_coint_tax);
Varg = (I_coint_spending)' * Sig * Sig' * (I_coint_spending);
VarpdT = ((I_pi + I_gdp + I_dt)' * Psi - (I_y1 + I_yspr)') * inv(eye(N) - k1x * Psi) * Sig * Sig' * (((I_pi + I_gdp + I_dt)' * Psi - (I_y1 + I_yspr)') * inv(eye(N) - k1x * Psi))';
CovpdT_tau = (I_coint_tax)' * Sig * Sig' * (((I_pi + I_gdp + I_dt)' * Psi - (I_y1 + I_yspr)') * inv(eye(N) - k1x * Psi))';
VarpdG = ((I_pi + I_gdp + I_dg)' * Psi - (I_y1 + I_yspr)') * inv(eye(N) - k1x * Psi) * Sig * Sig' * (((I_pi + I_gdp + I_dg)' * Psi - (I_y1 + I_yspr)') * inv(eye(N) - k1x * Psi))';
CovpdG_g = (I_coint_spending)' * Sig * Sig' * (((I_pi + I_gdp + I_dg)' * Psi - (I_y1 + I_yspr)') * inv(eye(N) - k1x * Psi))';

EPDT = exp(EpdT + Etau + 0.5 * Vartau + 0.5 * VarpdT + CovpdT_tau);
EPDG = exp(EpdG + Eg + 0.5 * Varg + 0.5 * VarpdG + CovpdG_g);

WedgeT = exp(pdT(2:end)) .* exp((I_coint_tax)' * X2t(2:end, :)') - EPDT;
WedgeG = exp(pdG(2:end)) .* exp((I_coint_spending)' * X2t(2:end, :)') - EPDG;

% Innovation in debt
debtgdp = gdebt(2:end);
deltalogd = [log(debtgdp(1)) - log(gdebt(1)), log(debtgdp(2:end)) - log(debtgdp(1:(end - 1)))];
d0 = mean(deltalogd);
coint_debt = 12;
X2 = [X2 log(debtgdp)' - mean(log(debtgdp))];
X2t = X2;

X2t(:, coint_tax) = log(taxrevgdp) - mean(log(taxrevgdp));
X2t(:, coint_spending) = log(spendgdp) - mean(log(spendgdp));

z_var = X2(2:end, :);
z_varlag = X2(1:end - 1, :);
Y = z_var;
X = z_varlag;

regr = ols(Y(:, coint_debt), [ones(T - 1, 1), X(:, :)]);
Psi_debt = regr.beta(2:end)';
Edebt = Psi_debt * (X2t(1:end - 1, :)');
WedgeDebt = X2t(2:end, coint_debt)' - Edebt;

ttime_date = [];

for i = 1948:2020
    ttime_date = [ttime_date; datenum(i, 12, 31)];
end

ttime_date = ttime_date(1:(end));

f = figure;
plot(ttime_date, WedgeT - WedgeG, 'b', 'LineWidth', 3)
hold on
plot(ttime_date, WedgeDebt, 'r', 'LineWidth', 3)
legend('PV(Surplus)/GDP - E(PV(Surplus)/GDP)', 'Debt/GDP - E(Debt/GDP)', 'Location', 'Northeast');

xticks([datenum(1950:10:2020, 1, 1)]);
datetick('x', 'yyyy', 'keepticks');
xlim([datenum(1948, 1, 1) datenum(2020, 12, 31)])
ylabel('%')
tmp = ylim;
p = recessionplot;
ylim(tmp);

for i = 1:12
    set(get(get(p(i), 'Annotation'), 'LegendInformation'), 'IconDisplayStyle', 'off');
end

grid;
set(gca, 'FontName', 'Times New Roman')
set(gca, 'FontSize', 12);
f.PaperSize = [6 4];
print(['../figures/pic-wedge_cyclicality_CS'], '-dpdf', '-fillpage');
