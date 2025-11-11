clear; close all;

load('../ConvYield/MAT/convyield.mat');
cy = m.spread(2:end);
gdebt = xlsread('../RawData/nominal_aggr_debt_1946_2020_annual.xlsx', 'Sheet1', 'd2:d76')';
cy_gdp_a = gdebt' .* m.mult .* m.spread;
cy_gdp_a = cy_gdp_a(2:end);

ttime = [1947:2020];

ttime_date = [];

for i = 1947:2020
    ttime_date = [ttime_date; datenum(i, 12, 31)];
end

ttime_date = ttime_date(1:(end));

f = figure;
plot(ttime_date, cy * 100, 'b', 'LineWidth', 3)
xticks([datenum(1950:10:2020, 1, 1)]);
datetick('x', 'yyyy', 'keepticks');
xlim([datenum(1947, 1, 1) datenum(2020, 12, 31)])
ylabel('Convenience Yield (%)')
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
print('../figures/pic-cy', '-dpdf', '-fillpage');

f = figure;
plot(ttime_date, cy_gdp_a * 100, 'b', 'LineWidth', 3)
xticks([datenum(1950:10:2020, 1, 1)]);
datetick('x', 'yyyy', 'keepticks');
xlim([datenum(1947, 1, 1) datenum(2020, 12, 31)])
ylabel('Seigniorage Revenue/GDP(%)')
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
print('../figures/pic-sgnrev', '-dpdf', '-fillpage');
