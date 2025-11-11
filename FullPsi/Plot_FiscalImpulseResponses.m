%% This code plots Fiscal Impulse Responses
clear; close all;

LoadData_Benchmark;

shocks = [gdppos, dtpos, dgpos];
shock_size = [-0.01, log(1 - 0.01 / mean(taxrevgdp)), log(1 + 0.01 / mean(spendgdp))];
lent = 16;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Simulation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
nround = 1e4;

z1_coint1pos = zeros(lent - 1, nround);
z1_coint2pos = zeros(lent - 1, nround);
z1_spos = zeros(lent - 1, nround);

sc = parallel.pool.Constant(RandStream('Threefry'));
parfor iround = 1:nround
    stream = sc.Value;
    stream.Substream = iround;

    X1 = zeros(size(X2));
    shock_rng = randi(stream, [1, T - 1], 1, T);

    for t = 2:T
        X1(t, :) = X1(t - 1, :) * Psi' + eps(shock_rng(t), :);
    end

    [Psi1, Sig1] = run_VAR(X1, N, T, inflpos, y1pos, ysprpos, gdppos, divgrpos, pdpos, dtpos, dgpos, ...
        coint_tax, coint_spending, coint_div);

    z1 = zeros(lent, N);
    z1(2, :) = Sig(:, shocks(1))' / Sig(shocks(1), shocks(1)) * shock_size(1);

    for t = 3:lent
        z1(t, :) = Psi1 * z1(t - 1, :)';
    end

    spos = N + 1;
    z1(:, spos) = mean(taxrevgdp) * exp(z1(:, coint_tax)) - mean(spendgdp) * exp(z1(:, coint_spending));

    z1_coint1pos(:, iround) = (mean(taxrevgdp) * (exp(z1(2:lent, coint_tax)) - 1)) * 100;
    z1_coint2pos(:, iround) = (mean(spendgdp) * (exp(z1(2:lent, coint_spending)) - 1)) * 100;
    z1_spos(:, iround) = (z1(2:lent, spos) - z1(1, spos)) * 100;

    z2 = zeros(lent, N);
    z2(2, :) = Sig(:, shocks(2))' / Sig(shocks(2), shocks(2)) * shock_size(2);

    for t = 3:lent
        z2(t, :) = Psi1 * z2(t - 1, :)';
    end

    spos = N + 1;
    z2(:, spos) = mean(taxrevgdp) * exp(z2(:, coint_tax)) - mean(spendgdp) * exp(z2(:, coint_spending));

    z2_coint1pos(:, iround) = (mean(taxrevgdp) * (exp(z2(2:lent, coint_tax)) - 1)) * 100;
    z2_coint2pos(:, iround) = (mean(spendgdp) * (exp(z2(2:lent, coint_spending)) - 1)) * 100;
    z2_spos(:, iround) = (z2(2:lent, spos) - z2(1, spos)) * 100;

    z3 = zeros(lent, N);
    z3(2, :) = Sig(:, shocks(3))' / Sig(shocks(3), shocks(3)) * shock_size(3);

    for t = 3:lent
        z3(t, :) = Psi1 * z3(t - 1, :)';
    end

    spos = N + 1;
    z3(:, spos) = mean(taxrevgdp) * exp(z3(:, coint_tax)) - mean(spendgdp) * exp(z3(:, coint_spending));

    z3_coint1pos(:, iround) = (mean(taxrevgdp) * (exp(z3(2:lent, coint_tax)) - 1)) * 100;
    z3_coint2pos(:, iround) = (mean(spendgdp) * (exp(z3(2:lent, coint_spending)) - 1)) * 100;
    z3_spos(:, iround) = (z3(2:lent, spos) - z3(1, spos)) * 100;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% IRF
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
zname = ['x', 't', 'g'];

scales = [.3, 1.2, 1.2];

for i = 1:3

    if i == 1
        z_coint1pos = z1_coint1pos;
        z_coint2pos = z1_coint2pos;
        z_spos = z1_spos;
    elseif i == 2
        z_coint1pos = z2_coint1pos;
        z_coint2pos = z2_coint2pos;
        z_spos = z2_spos;
    elseif i == 3
        z_coint1pos = z3_coint1pos;
        z_coint2pos = z3_coint2pos;
        z_spos = z3_spos;
    end

    f = figure;

    z = zeros(lent, N);
    z(2, :) = Sig(:, shocks(i))' / Sig(shocks(i), shocks(i)) * shock_size(i);

    for t = 3:lent
        z(t, :) = Psi * z(t - 1, :)';
    end

    spos = N + 1;
    z(:, spos) = mean(taxrevgdp) * exp(z(:, coint_tax)) - mean(spendgdp) * exp(z(:, coint_spending));

    subplot(1, 3, 1);
    plot(1:lent - 1, zeros(1, lent - 1), 'k');
    hold on
    plot(1:lent - 1, (mean(taxrevgdp) * (exp(z(2:lent, coint_tax)) - 1)) * 100, 'b', 'LineWidth', 3)
    ylim([-scales(i), scales(i)]);
    xlim([1, lent - 1]);
    ylabel('Tax/GDP (%)');
    set(gca, 'Layer', 'top', 'FontSize', 9)
    set(gca, 'FontName', 'Times New Roman')
    grid

    hold on
    err1 = [prctile(z_coint1pos', normcdf(1) * 100); prctile(z_coint1pos', normcdf(-1) * 100)];
    err2 = [prctile(z_coint1pos', normcdf(2) * 100); prctile(z_coint1pos', normcdf(-2) * 100)];
    patch([1:lent - 1, fliplr(1:lent - 1)], [err1(1, :), fliplr(err1(2, :))], 'b', ...
        'FaceAlpha', 0.1, 'EdgeColor', 'none') % Shaded Confidence Intervals
    patch([1:lent - 1, fliplr(1:lent - 1)], [err2(1, :), fliplr(err2(2, :))], [61, 157, 214] / 255, ...
        'FaceAlpha', 0.1, 'EdgeColor', 'none') % Shaded Confidence Intervals
    hold off

    subplot(1, 3, 2);
    plot(1:lent - 1, zeros(1, lent - 1), 'k');
    hold on
    plot(1:lent - 1, (mean(spendgdp) * (exp(z(2:lent, coint_spending)) - 1)) * 100, 'b', 'LineWidth', 3)
    ylim([-scales(i), scales(i)]);
    xlim([1, lent - 1]);
    ylabel('Spending/GDP (%)');
    set(gca, 'Layer', 'top', 'FontSize', 9)
    set(gca, 'FontName', 'Times New Roman')
    grid

    hold on
    err1 = [prctile(z_coint2pos', normcdf(1) * 100); prctile(z_coint2pos', normcdf(-1) * 100)];
    err2 = [prctile(z_coint2pos', normcdf(2) * 100); prctile(z_coint2pos', normcdf(-2) * 100)];
    patch([1:lent - 1, fliplr(1:lent - 1)], [err1(1, :), fliplr(err1(2, :))], 'b', ...
        'FaceAlpha', 0.1, 'EdgeColor', 'none') % Shaded Confidence Intervals
    patch([1:lent - 1, fliplr(1:lent - 1)], [err2(1, :), fliplr(err2(2, :))], [61, 157, 214] / 255, ...
        'FaceAlpha', 0.1, 'EdgeColor', 'none') % Shaded Confidence Intervals
    hold off

    subplot(1, 3, 3);
    plot(1:lent - 1, zeros(1, lent - 1), 'k');
    hold on
    plot(1:lent - 1, (z(2:lent, spos) - z(1, spos)) * 100, 'b', 'LineWidth', 3)
    ylim([-scales(i), scales(i)]);
    xlim([1, lent - 1]);
    ylabel('Surplus/GDP (%)');
    set(gca, 'Layer', 'top', 'FontSize', 9)
    set(gca, 'FontName', 'Times New Roman')
    grid

    hold on
    err1 = [prctile(z_spos', normcdf(1) * 100); prctile(z_spos', normcdf(-1) * 100)];
    err2 = [prctile(z_spos', normcdf(2) * 100); prctile(z_spos', normcdf(-2) * 100)];
    patch([1:lent - 1, fliplr(1:lent - 1)], [err1(1, :), fliplr(err1(2, :))], 'b', ...
        'FaceAlpha', 0.1, 'EdgeColor', 'none') % Shaded Confidence Intervals
    patch([1:lent - 1, fliplr(1:lent - 1)], [err2(1, :), fliplr(err2(2, :))], [61, 157, 214] / 255, ...
        'FaceAlpha', 0.1, 'EdgeColor', 'none') % Shaded Confidence Intervals
    hold off

    f.PaperSize = [12 3];

    switch i
        case 1
            print('../figures/pic-3CI_x_annual', '-dpdf', '-fillpage');
        case 2
            print('../figures/pic-3CI_dtau_annual', '-dpdf', '-fillpage');
        case 3
            print('../figures/pic-3CI_dg_annual', '-dpdf', '-fillpage');
    end

end
