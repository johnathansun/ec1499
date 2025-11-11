function aux_plot_CI(ttime, s, std_coeff, gdebt, upper)

    y1 = s;
    y2 = s + 2 * std_coeff;
    y3 = s - 2 * std_coeff;

    % plot the area
    y = [y3' (y2' - y3')];
    ha = area(ttime, y); hold on;
    set(ha(1), 'FaceColor', 'none') % this makes the bottom area invisible
    set(ha, 'LineStyle', 'none')

    % plot the line edges
    hb = plot(ttime, y3, 'LineWidth', 1); hold on;
    hc = plot(ttime, y2, 'LineWidth', 1); hold on;

    % plot the point estimates
    hd = plot(ttime, y1, 'LineWidth', 3); hold on;
    set(ha(2), 'FaceColor', [0.7 0.7 0.7]);
    set(hb, 'Color', [0.9 0.9 0.9]);
    set(hc, 'Color', [0.9 0.9 0.9]);
    set(hd, 'Color', 'r');

    y1 = s;
    y2 = s + 1 * std_coeff;
    y3 = s - 1 * std_coeff;

    % plot the area
    y = [y3' (y2' - y3')];
    ha = area(ttime, y); hold on;
    set(ha(1), 'FaceColor', 'none') % this makes the bottom area invisible
    set(ha, 'LineStyle', 'none')

    % plot the line edges
    hb = plot(ttime, y3, 'LineWidth', 1); hold on;
    hc = plot(ttime, y2, 'LineWidth', 1); hold on;

    % plot the point estimates
    hd = plot(ttime, y1, 'LineWidth', 3); hold on;
    set(ha(2), 'FaceColor', [0.5 0.5 0.5]);
    set(hb, 'Color', [0.9 0.9 0.9]);
    set(hc, 'Color', [0.9 0.9 0.9]);
    set(hd, 'Color', 'red');

    hg = plot(ttime, gdebt(2:end), 'Color', 'blue', 'LineWidth', 3); hold on;
    hu = plot(ttime, upper * ones(1, size(ttime, 2)), 'Color', 'black', 'LineWidth', 2); hold on;

    axis([min(ttime) max(ttime) -1.5 2.5])

    hleglines = [hd(1) hg(1) hu(1)];
    legend(hleglines, 'PV(Surplus)/GDP', 'Debt/GDP', 'Upper Bound at z=0')

    xlabel('Year')
    ylabel('PV(Surplus)/GDP')
    set(gca, 'Layer', 'top', 'FontSize', 9)
    set(gca, 'FontName', 'Times New Roman')

    grid
