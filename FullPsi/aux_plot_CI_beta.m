function aux_plot_CI_beta(ttime, s, std_coeff)

    y1 = s;
    y2 = s + 2 * std_coeff;
    y3 = s - 2 * std_coeff;

    % plot the area
    y = [y3' (y2' - y3')];
    ha = area(ttime, y); hold on;
    set(ha(1), 'FaceColor', 'none')
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
    set(ha(1), 'FaceColor', 'none')
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

    axis([min(ttime) max(ttime) -2 3])

    xlabel('Horizon in yrs')
    ylabel('GDP Growth Beta')
    set(gca, 'Layer', 'top', 'FontSize', 9)
    set(gca, 'FontName', 'Times New Roman')

    grid
