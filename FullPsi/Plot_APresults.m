ttime = [1947:2020];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Bond yields
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

f = figure;
subplot(2, 3, 1)
plot(ttime, 100 * (-Api(1) / 1 - Bpi(:, 1)' / 1 * X2t')', 'b', ttime, 100 * ynom1, 'r--', 'LineWidth', 3)
xlim([1945, 2020]); ylim([0, 15]);
legend('model', 'data', 'Location', 'NorthWest')
ylabel('% per year')
set(gca, 'FontName', 'Times New Roman')
set(gca, 'FontSize', 12);
title('Nom yield on 1-yr bond')
grid
subplot(2, 3, 2)
plot(ttime, 100 * (-Api(2) / 2 - Bpi(:, 2)' / 2 * X2t')', 'b', ttime, 100 * ynom2, 'r--', 'LineWidth', 3)
xlim([1945, 2020]); ylim([0, 15]);
legend('model', 'data', 'Location', 'NorthWest')
ylabel('% per year')
set(gca, 'FontName', 'Times New Roman')
set(gca, 'FontSize', 12);
title('Nom yield on 2-yr bond')
grid
subplot(2, 3, 3)
plot(ttime, 100 * (-Api(5)' ./ 5 -Bpi(:, 5)' ./ 5 * X2t')', 'b', ttime, 100 * ynom5, 'r--', 'LineWidth', 3)
xlim([1945, 2020]); ylim([0, 15]);
legend('model', 'data', 'Location', 'NorthWest')
ylabel('% per year')
set(gca, 'FontName', 'Times New Roman')
set(gca, 'FontSize', 12);
title('Nom yield on 5-yr bond')
grid
subplot(2, 3, 4)
plot(ttime, 100 * (-Api(10) / 10 - Bpi(:, 10)' / 10 * X2t')', 'b', ttime, 100 * ynom10, 'r--', 'LineWidth', 3)
xlim([1945, 2020]); ylim([0, 15]);
legend('model', 'data', 'Location', 'NorthWest')
ylabel('% per year')
set(gca, 'FontName', 'Times New Roman')
set(gca, 'FontSize', 12);
title('Nom yield on 10-yr bond')
grid
subplot(2, 3, 5)
plot(ttime, 100 * (-Api(20) / 20 - Bpi(:, 20)' / 20 * X2t')', 'b', ttime, 100 * ynom20, 'r--', 'LineWidth', 3)
xlim([1945, 2020]); ylim([0, 15]);
legend('model', 'data', 'Location', 'NorthWest')
ylabel('% per year')
set(gca, 'FontName', 'Times New Roman')
set(gca, 'FontSize', 12);
title('Nom yield on 20-yr bond')
grid
subplot(2, 3, 6)
plot(ttime, 100 * (-Api(30)' ./ 30 -Bpi(:, 30)' ./ 30 * X2t')', 'b', ttime, 100 * ynom30, 'r--', 'LineWidth', 3)
xlim([1945, 2020]); ylim([0, 15]);
legend('model', 'data', 'Location', 'NorthWest')
ylabel('% per year')
title('Nom yield on 30-yr bond')
grid
set(gca, 'FontName', 'Times New Roman')
set(gca, 'FontSize', 12);

f.PaperSize = [6 * 3 4 * 2];
print(['../figures/pic-yields-step', num2str(step)], '-dpdf', '-fillpage');

% real
f = figure;
ttime = [1990:2020];
subplot(2, 3, 1)
plot(ttime, 100 * (-A(5) / 5 - B(:, 5)' / 5 * X2t(end - 30:end, :)')', 'b', ttime, 100 * tipsdata(end - 30:end, 1), 'r--', 'LineWidth', 3)
xlim([1988, 2020]); ylim([-3, 6]);
legend('model', 'data', 'Location', 'SouthWest')
ylabel('% per year')
title('Real yield on 5-yr TIPS')
set(gca, 'FontName', 'Times New Roman')
set(gca, 'FontSize', 12);
grid
subplot(2, 3, 2)
plot(ttime, 100 * (-A(7) / 7 - B(:, 7)' / 7 * X2t(end - 30:end, :)')', 'b', ttime, 100 * tipsdata(end - 30:end, 2), 'r--', 'LineWidth', 3)
xlim([1988, 2020]); ylim([-3, 6]);
legend('model', 'data', 'Location', 'SouthWest')
ylabel('% per year')
title('Real yield on 7-yr TIPS')
set(gca, 'FontName', 'Times New Roman')
set(gca, 'FontSize', 12);
grid
subplot(2, 3, 3)
plot(ttime, 100 * (-A(10)' ./ 10 -B(:, 10)' ./ 10 * X2t(end - 30:end, :)')', 'b', ttime, 100 * tipsdata(end - 30:end, 3), 'r--', 'LineWidth', 3)
xlim([1988, 2020]); ylim([-3, 6]);
legend('model', 'data', 'Location', 'SouthWest')
title('Real yield on 10-yr TIPS')
ylabel('% per year')
set(gca, 'FontName', 'Times New Roman')
set(gca, 'FontSize', 12);
grid
subplot(2, 3, 4)
plot(ttime, 100 * (-A(20) / 20 - B(:, 20)' / 20 * X2t(end - 30:end, :)')', 'b', ttime, 100 * tipsdata(end - 30:end, 4), 'r--', 'LineWidth', 3)
xlim([1988, 2020]); ylim([-3, 6]);
legend('model', 'data', 'Location', 'SouthWest')
ylabel('% per year')
title('Real yield on 20-yr TIPS')
set(gca, 'FontName', 'Times New Roman')
set(gca, 'FontSize', 12);
grid
subplot(2, 3, 5)
plot(ttime, 100 * (-A(30)' ./ 30 -B(:, 30)' ./ 30 * X2t(end - 30:end, :)')', 'b', ttime, 100 * tipsdata(end - 30:end, 5), 'r--', 'LineWidth', 3)
xlim([1988, 2020]); ylim([-3, 6]);
legend('model', 'data', 'Location', 'SouthWest')
title('Real yield on 30-yr TIPS')
set(gca, 'FontName', 'Times New Roman')
set(gca, 'FontSize', 12);
ylabel('% per year')
grid

f.PaperSize = [6 * 3 4 * 2];
print(['../figures/pic-realyields-step', num2str(step)], '-dpdf', '-fillpage');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Bond RP
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

f = figure;
ttime = [1947:2020];
maxmat = 500;
subplot(2, 2, 1)
plot([1:1:maxmat], -100 * Api(1:maxmat) ./ [1:1:maxmat]', 'b', 'LineWidth', 3)
xlim([0, maxmat])
ylim([0, 12])
title('Average nominal yield curve model')
ylabel('percent per annum')
xlabel('maturity in years')
set(gca, 'FontName', 'Times New Roman')
set(gca, 'FontSize', 12);
grid
subplot(2, 2, 2)
plot([1:1:maxmat], -100 * A(1:maxmat) ./ [1:1:maxmat]', 'b', 'LineWidth', 3)
xlim([0, maxmat])
ylim([0, 4])
title('Average real yield curve model')
ylabel('percent per annum')
xlabel('maturity in years')
set(gca, 'FontName', 'Times New Roman')
set(gca, 'FontSize', 12);
grid
subplot(2, 2, 3)
plot(ttime, 100 * bondriskpremTV_model', 'b', ttime, 100 * bondriskpremTV_data', 'r', 'LineWidth', 3)
legend('model', 'data', 'Location', 'NorthWest')
ylabel('% per year')
set(gca, 'FontName', 'Times New Roman')
set(gca, 'FontSize', 12);
xlim([1945, 2020]); ylim([-2, 5]);
title('Risk Premium on 5-yr Nominal Bond')
grid
subplot(2, 2, 4)
plot(ttime, 100 * ynom5_model, 'b', ttime, 100 * yreal5_model, 'r--', ttime, 100 * expinfl5, 'k:', ttime, 100 * inflbrp(5, :)', 'm-.', 'LineWidth', 3)
legend('ynom', 'yreal', 'exp infl', 'IRP')
title('Decomposing Nominal Yield on 5-Year Bond')
xlim([1945, 2020]); ylim([-4, 15]);
set(gca, 'FontName', 'Times New Roman')
set(gca, 'FontSize', 12);
grid

f.PaperSize = [6 * 2 4 * 2];
print(['../figures/pic-avgyieldbrp-step', num2str(step)], '-dpdf', '-fillpage');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Equity RP and PD ratio
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

f = figure;
subplot(1, 2, 1)
plot(ttime, equityriskpremTV_model', 'b', 'LineWidth', 3);
hold on
plot(ttime, equityriskpremTV_data', 'r--', 'LineWidth', 3)
xlim([1945, 2020]); ylim([-15, 40]);
legend('model', 'data', 'Location', 'Northeast')
ylabel('% per year')
set(gca, 'FontName', 'Times New Roman')
set(gca, 'FontSize', 14);
title('Equity risk premium')
grid
subplot(1, 2, 2)
plot(ttime, PDm_model', 'b', 'LineWidth', 3)
hold on
plot(ttime, (exp(A0m + I_pdm' * X2t'))', 'r--', 'LineWidth', 3)
xlim([1945, 2020]); ylim([0, 95]);
legend('model', 'data', 'Location', 'NorthWest')
set(gca, 'FontName', 'Times New Roman')
set(gca, 'FontSize', 14);
title('Price-Dividend Ratio on Equity')
grid

f.PaperSize = [6 * 2 4];
print(['../figures/pic-equitypd-step', num2str(step)], '-dpdf', '-fillpage');


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Puzzle
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

PDg_model0 = sum(exp(Ag));
PDt_model0 = sum(exp(At));
PDx_model0 = sum(exp(Ax));
pv0 = (PDt_model0) .* exp(mean(log(taxrevgdp))) - (PDg_model0) .* exp(mean(log(spendgdp)));

f = figure;
plot(ttime, ((PDt_model) .* taxrevgdp - (PDg_model) .* spendgdp), ...
    ttime, pv0 * ones(1, T), ':', ...
    ttime, gdebt(end - Tend + 1:end)', '--', 'LineWidth', 1.2);
xlim([min(ttime), max(ttime)])
legend('PV(Surplus)/GDP', 'Steady State Value', 'Govt Debt Outstanding/GDP', 'location', 'south')
ylabel('PV/GDP Ratio')
set(gca, 'FontName', 'Times New Roman')
set(gca, 'FontSize', 12);
grid
f.PaperSize = [6 4];
print(['../figures/pic-puzzle-step', num2str(step)], '-dpdf', '-fillpage');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% RP term structure
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

horizon = 100;

f = figure;
plot(1:horizon, (-Ag(1:horizon) ./ (1:horizon) - y0nom_1 + (g0 + x0 + pi0)) * 100, 'LineWidth', 1.5)
hold on
plot(1:horizon, (-At(1:horizon) ./ (1:horizon) - y0nom_1 + (tau0 + x0 + pi0)) * 100, '--', 'LineWidth', 1.5)
hold on
plot(1:horizon, (-Am(1:horizon)' ./ (1:horizon) - y0nom_1 + (mu_m + x0 + pi0)) * 100, ':black', 'LineWidth', 1.5)
hold on
plot(1:horizon, (-Ax(1:horizon) ./ (1:horizon) - y0nom_1 + (x0 + pi0)) * 100, 'Color', [0 0.5 0], 'LineStyle', '-.', 'LineWidth', 1.5)
hold on
plot(1:horizon, (-Api(1:horizon)' ./ (1:horizon) - y0nom_1) * 100, 'Color', [0 0 0.5], 'LineStyle', '-.', 'LineWidth', 1.5)

hold off
% ylim([-6 12])
set(gca, 'FontName', 'Times New Roman')
set(gca, 'FontSize', 12);
legend('G Claim', 'T Claim', 'Equity', 'GDP', 'Nominal Bond', 'Location', 'Southeast')
xlabel('Period (Year)')
ylabel('Risk Premium %')
grid
f.PaperSize = [6 4];
print(['../figures/pic-rptermstr-step', num2str(step)], '-dpdf', '-fillpage');
