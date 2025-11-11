clear; close all;

f = figure;

load('MAT/CS30.mat');
plot(ttime, s, 'r', 'LineWidth', 3); hold on;
axis([min(ttime) max(ttime) -4 4])

load('MAT/CS_ConvYield.mat');
plot(ttime, s, 'k', 'LineWidth', 3); hold on;

load('MAT/CS_WithDebt30.mat');
plot(ttime, s, 'g', 'LineWidth', 3); hold on;

load('MAT/CS_LongSample30.mat');
plot(ttime, s, 'r--', 'LineWidth', 3); hold on;

load('MAT/CS30.mat');
plot(ttime, gdebt(2:end), 'b', 'LineWidth', 3); hold on;

legend('Benchmark', 'with Conv Yield', 'Debt in VAR', 'Long Sample', ...
    'Debt/GDP', 'Location', 'NorthWest')

xlabel('Year')
ylabel('PV(Surplus)/GDP')
set(gca, 'Layer', 'top', 'FontSize', 9)
set(gca, 'FontName', 'Times New Roman')
ylim([-1.5 1.5]);
grid
f.PaperSize = [6 4];
print(['../figures/cs_summary.pdf'], '-dpdf', '-fillpage');
