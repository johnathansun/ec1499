clear; close all;

load('MAT/result_step3.mat');
ttime = [1947:2020];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% fig.1 U.S. Government Surplus
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Figure for surplus
ttime_date = [];

for i = 1947:2020
    ttime_date = [ttime_date; datenum(i, 12, 31)];
end

ttime_date = ttime_date(1:(end));

f = figure;
plot(ttime_date, surplusgdp * 100, 'b', 'LineWidth', 3)
xticks([datenum(1950:10:2020, 1, 1)]);
datetick('x', 'yyyy', 'keepticks');
xlim([datenum(1947, 1, 1) datenum(2020, 12, 31)])
ylabel('Surplus/GDP (%)')
tmp = ylim;
p = recessionplot;
ylim(tmp);

for i = 1:11
    set(get(get(p(i), 'Annotation'), 'LegendInformation'), 'IconDisplayStyle', 'off');
end

tmp = ylim;
p = recessionplot('recessions', [datenum(2020, 3, 1) datenum(2020, 4, 30); ]);
ylim(tmp);
set(get(get(p(1), 'Annotation'), 'LegendInformation'), 'IconDisplayStyle', 'off');
grid;
set(gca, 'FontName', 'Times New Roman')
set(gca, 'FontSize', 12);
f.PaperSize = [6 4];
print('../figures/pic-surplus_2020', '-dpdf', '-fillpage');

f = figure;
plot(ttime_date, taxrevgdp * 100, 'b', ttime_date, spendgdp * 100, 'r--', 'LineWidth', 3)
xticks([datenum(1950:10:2020, 1, 1)]);
datetick('x', 'yyyy', 'keepticks');
xlim([datenum(1947, 1, 1) datenum(2020, 12, 31)])
ylabel('Tax or Spending/GDP (%)')
tmp = ylim;
p = recessionplot;
ylim(tmp);

for i = 1:11
    set(get(get(p(i), 'Annotation'), 'LegendInformation'), 'IconDisplayStyle', 'off');
end

tmp = ylim;
p = recessionplot('recessions', [datenum(2020, 3, 1) datenum(2020, 4, 30); ]);
ylim(tmp);
set(get(get(p(1), 'Annotation'), 'LegendInformation'), 'IconDisplayStyle', 'off');
grid;
legend('Tax/GDP', 'Spending/GDP')
set(gca, 'FontName', 'Times New Roman')
set(gca, 'FontSize', 12);
f.PaperSize = [6 4];
print('../figures/pic-taxspend_2020', '-dpdf', '-fillpage');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% fig.6 Valuations of Government Cash Flows
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Lt = L0 + L1 * X2';
sqrt(diag(Lt' * Lt));
sqrt(exp(diag(Lt' * Lt)) - 1);

PDg_model0 = sum(exp(Ag));
PDt_model0 = sum(exp(At));
PDx_model0 = sum(exp(Ax));
pv0 = (PDt_model0) .* exp(mean(log(taxrevgdp))) - (PDg_model0) .* exp(mean(log(spendgdp)));

f = figure;

plot(1947 - 1 + (1:T), PDt_model, 'b', ...
    1947 - 1 + (1:T), PDg_model, 'r--', 'LineWidth', 3)
hold on;
plot(1947 - 1 + (1:T), PDt_model0, 'b.', ...
    1947 - 1 + (1:T), PDg_model0, 'r.', 'LineWidth', 1)
xlim([1947, 2020])
legend('PV(Tax)/Tax', 'PV(Spending)/Spending', 'Location', 'northwest');
ylabel('PV/Cash Flow Ratio')
grid

set(gca, 'FontName', 'Times New Roman')
set(gca, 'FontSize', 12);
f.PaperSize = [6 4];
print('../figures/pd_ratio', '-dpdf', '-fillpage');

f = figure;

plot(1947 - 1 + (1:T), (PDt_model) .* taxrevgdp, 'b', ...
    1947 - 1 + (1:T), (PDg_model) .* spendgdp, 'r--', 'LineWidth', 3)
xlim([1947, 2020])
hold on;
plot(1947 - 1 + (1:T), (PDt_model0) * mean(taxrevgdp), 'b.', ...
    1947 - 1 + (1:T), (PDg_model0) * mean(spendgdp), 'r.', 'LineWidth', 1)
legend('PV(Tax)/GDP', 'PV(Spending)/GDP', 'Location', 'northwest');
ylabel('PV/GDP Ratio')
grid

set(gca, 'FontName', 'Times New Roman')
set(gca, 'FontSize', 12);
f.PaperSize = [6 4];
print('../figures/pgdp_ratio', '-dpdf', '-fillpage');


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% fig.7 Present Value of Government Surpluses and Market Value of Government Debt
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% add the result with conv yield
load('../FullPsi_ConvYield/MAT/result_pv.mat', 'pvs_withcy');

f = figure;
plot(ttime, ((1 + PDt_model) .* taxrevgdp - (1 + PDg_model) .* spendgdp), 'b', ...
    ttime, pvs_withcy, 'g', ...
    ttime, pv0 * ones(1, T), 'r--', ...
    ttime, gdebt(2:end)', 'k:', 'LineWidth', 3);
xlim([1947, 2020])
legend('PV(Surplus)/GDP', 'PV(Surplus+Conv Yield)/GDP', 'Steady State Value', 'Govt Debt Outstanding/GDP', 'location', 'south')
ylabel('PV/GDP Ratio')
grid
set(gca, 'FontName', 'Times New Roman')
set(gca, 'FontSize', 12);
f.PaperSize = [6 4];
print('../figures/puzzle_actualCF', '-dpdf', '-fillpage');

pvgdp0 = ((PDt_model0) .* exp(mean(log(taxrevgdp))) - (PDg_model0) .* exp(mean(log(spendgdp)))) ./ ...
    (1 + PDx_model0);

f = figure;
plot(ttime, ((PDt_model) .* taxrevgdp - (PDg_model) .* spendgdp) ./ (1 + PDx_model), 'b', ...
    ttime, pvgdp0 * ones(1, T), 'r--', ...
    ttime, gdebt(2:end)' ./ (1 + PDx_model), 'k:', 'LineWidth', 3);
xlim([1947, 2020])
legend('PV(Surplus)/PV(GDP)', 'Steady State Value', 'Govt Debt Outstanding/PV(GDP)', 'location', 'south')
ylabel('PV/PV(GDP) Ratio')
set(gca, 'FontName', 'Times New Roman')
set(gca, 'FontSize', 12);
f.PaperSize = [6 4];
grid
print('../figures/puzzle_actualCF_GDP', '-dpdf', '-fillpage');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% fig.8 Term Structure of Risk Premia on the T-Claim and the G-Claim
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

horizon = 50;

f = figure;
plot(1:horizon, (-Ag(1:horizon) ./ (1:horizon) - y0nom_1 + (g0 + x0 + pi0)) * 100, 'b', 'LineWidth', 3)
hold on
plot(1:horizon, (-At(1:horizon) ./ (1:horizon) - y0nom_1 + (tau0 + x0 + pi0)) * 100, 'r--', 'LineWidth', 3)
hold on
plot(1:horizon, (-Am(1:horizon)' ./ (1:horizon) - y0nom_1 + (mu_m + x0 + pi0)) * 100, 'k:', 'LineWidth', 3)
hold on
plot(1:horizon, (-Ax(1:horizon) ./ (1:horizon) - y0nom_1 + (x0 + pi0)) * 100, 'g', 'LineStyle', '-.', 'LineWidth', 3)
hold on
plot(1:horizon, (-Api(1:horizon)' ./ (1:horizon) - y0nom_1) * 100, 'm', 'LineStyle', '-.', 'LineWidth', 3)

hold off
set(gca, 'FontName', 'Times New Roman')
set(gca, 'FontSize', 12);
legend('G Claim', 'T Claim', 'Equity', 'GDP', 'Nominal Bond', 'Location', 'Southeast')
xlabel('Period (Year)')
ylabel('Risk Premium (%)')
grid
f.PaperSize = [6 4];
print('../figures/cum_rp_term_annual', '-dpdf', '-fillpage');
