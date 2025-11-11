clear; close all;

[data, txt] = xlsread('../RawData/MaindatafileA_Feb2021_longsample.xlsx', 'VAR2');

debt_gdp = data(19:92, 16);
gdp_growth = data(19:92, 5);
surplus = data(19:92, 4);
g_ratio = data(19:92, 3);
t_ratio = data(19:92, 2);

div_growth_raw = data(19:92, 15);
inflation = data(19:92, 6);
div_ratio_growth = div_growth_raw - inflation - gdp_growth;
div_ratio = exp(cumsum(div_ratio_growth));

horizon_total = 50;

T = size(g_ratio, 1);

t_growth = zeros(T, horizon_total);
g_growth = zeros(T, horizon_total);
div_growth = zeros(T, horizon_total);
T_growth = zeros(T, horizon_total);
G_growth = zeros(T, horizon_total);
Div_growth = zeros(T, horizon_total);
GDP_growth = zeros(T, horizon_total);

for horizon = 1:horizon_total
    t_growth(1 + horizon:end, horizon) = log(t_ratio(1 + horizon:end, 1)) - log(t_ratio(1:end - horizon, 1));
    g_growth(1 + horizon:end, horizon) = log(g_ratio(1 + horizon:end, 1)) - log(g_ratio(1:end - horizon, 1));
    div_growth(1 + horizon:end, horizon) = log(div_ratio(1 + horizon:end, 1)) - log(div_ratio(1:end - horizon, 1));

    tmp = movsum(gdp_growth, [horizon - 1 0]);
    GDP_growth(1 + horizon:end, horizon) = tmp(1 + horizon:end);

    T_growth(:, horizon) = GDP_growth(:, horizon) + t_growth(:, horizon);
    G_growth(:, horizon) = GDP_growth(:, horizon) + g_growth(:, horizon);
    Div_growth(:, horizon) = GDP_growth(:, horizon) + div_growth(:, horizon);
end

for horizon = 1:50
    Y_Beta = T_growth(1 + horizon:end, horizon);
    X_Beta = GDP_growth(1 + horizon:end, horizon);
    mdl{horizon} = LinearModel.fit([X_Beta(:, 1)], Y_Beta);

    if horizon < 5
        [EstCov, se, coeffic] = hac(X_Beta(:, 1), Y_Beta, 'type', 'HAC', 'bandwidth', horizon, 'weights', 'TR');
        tmp_se_hac = se(2);
    else
        tmp_se_hac = mdl{horizon}.Coefficients.SE(2);
    end

    tmp_se = mdl{horizon}.Coefficients.SE(2);

    k = horizon;
    coeff_T(1, k) = mdl{horizon}.Coefficients.Estimate(2);
    coeff_T(2, k) = tmp_se_hac;
    coeff_T(3, k) = mdl{horizon}.Rsquared.Ordinary;
end

for horizon = 1:50
    Y_Beta = G_growth(1 + horizon:end, horizon);
    X_Beta = GDP_growth(1 + horizon:end, horizon);
    mdl{horizon} = LinearModel.fit([X_Beta(:, 1)], Y_Beta);

    if horizon < 5
        [EstCov, se, coeffic] = hac(X_Beta(:, 1), Y_Beta, 'type', 'HAC', 'bandwidth', horizon, 'weights', 'TR');
        tmp_se_hac = se(2);
    else
        tmp_se_hac = mdl{horizon}.Coefficients.SE(2);
    end

    tmp_se = mdl{horizon}.Coefficients.SE(2);
    tmp_se_hac = se(2);
    k = horizon;
    coeff_G(1, k) = mdl{horizon}.Coefficients.Estimate(2);
    coeff_G(2, k) = tmp_se_hac;
    coeff_G(3, k) = mdl{horizon}.Rsquared.Ordinary;
end

for horizon = 1:50
    Y_Beta = Div_growth(1 + horizon:end, horizon);
    X_Beta = GDP_growth(1 + horizon:end, horizon);
    mdl{horizon} = LinearModel.fit([X_Beta(:, 1)], Y_Beta);

    if horizon < 5
        [EstCov, se, coeffic] = hac(X_Beta(:, 1), Y_Beta, 'type', 'HAC', 'bandwidth', horizon, 'weights', 'TR');
        tmp_se_hac = se(2);
    else
        tmp_se_hac = mdl{horizon}.Coefficients.SE(2);
    end

    tmp_se = mdl{horizon}.Coefficients.SE(2);
    tmp_se_hac = se(2);
    k = horizon;
    coeff_Div(1, k) = mdl{horizon}.Coefficients.Estimate(2);
    coeff_Div(2, k) = tmp_se_hac;
    coeff_Div(3, k) = mdl{horizon}.Rsquared.Ordinary;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% bootstrap
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Mdl_d = arima(2, 0, 0);
EstMdl_d = estimate(Mdl_d, log(debt_gdp))
phi = EstMdl_d.AR{1, 1};
var = EstMdl_d.Variance / (1 - phi ^ 2);
std_innov = sqrt(var);
[E_d, V_d] = infer(EstMdl_d, log(debt_gdp));

Mdl_g = arima(1, 0, 0);
EstMdl_g = estimate(Mdl_g, log(g_ratio))
[E_g, V_g] = infer(EstMdl_g, log(g_ratio));

Mdl_t = arima(1, 0, 0);
EstMdl_t = estimate(Mdl_t, log(t_ratio))
[E_t, V_t] = infer(EstMdl_t, log(t_ratio));

Mdl_div = arima(1, 0, 0);
EstMdl_div = estimate(Mdl_div, log(div_ratio))
[E_div, V_div] = infer(EstMdl_div, log(div_ratio));

Mdl_gdp = arima(1, 0, 0);
EstMdl_gdp = estimate(Mdl_gdp, gdp_growth)
[E_gdp, V_gdp] = infer(EstMdl_gdp, gdp_growth);

E_sim = [E_gdp E_g E_t E_d E_div];

Z_gdp = E_sim(:, 1) ./ sqrt(V_gdp);
Z_g = E_sim(:, 2) ./ sqrt(V_g);
Z_t = E_sim(:, 3) ./ sqrt(V_t);
Z_d = E_sim(:, 4) ./ sqrt(V_d);
Z_div = E_sim(:, 5) ./ sqrt(V_div);

nboot_total = 1e4;

sc = parallel.pool.Constant(RandStream('Threefry'));
parfor nboot = 1:nboot_total
    stream = sc.Value;
    stream.Substream = nboot;
    
    E_rnd = datasample(stream, E_sim, 74, 'Replace', true);
    Z_gdp = E_rnd(:, 1) ./ sqrt(V_gdp);
    Z_g = E_rnd(:, 2) ./ sqrt(V_g);
    Z_t = E_rnd(:, 3) ./ sqrt(V_t);
    Z_d = E_rnd(:, 4) ./ sqrt(V_d);
    Z_div = E_rnd(:, 5) ./ sqrt(V_div);

    [gdp_growth_sim, E, V] = filter(EstMdl_gdp, Z_gdp(2:end), 'Y0', gdp_growth(1));
    [lg_ratio_sim, E, V] = filter(EstMdl_g, Z_g(2:end), 'Y0', log(g_ratio(1)));
    [lt_ratio_sim, E, V] = filter(EstMdl_t, Z_t(2:end), 'Y0', log(t_ratio(1)));
    [ldebt_gdp_sim, E, V] = filter(EstMdl_d, Z_d(3:end), 'Y0', [log(debt_gdp(1)); log(debt_gdp(2))]);
    [ldiv_ratio_sim, E, V] = filter(EstMdl_div, Z_div(2:end), 'Y0', log(div_ratio(1)));

    gdp_growth_sim = [gdp_growth(1); gdp_growth_sim];
    t_ratio_sim = [t_ratio(1); exp(lt_ratio_sim)];
    g_ratio_sim = [g_ratio(1); exp(lg_ratio_sim)];
    debt_gdp_sim = [debt_gdp(1); debt_gdp(2); exp(ldebt_gdp_sim)];
    div_ratio_sim = [div_ratio(1); exp(ldiv_ratio_sim)];

    T = size(g_ratio, 1);
    T_growth = zeros(T, horizon_total);
    G_growth = zeros(T, horizon_total);
    GDP_growth = zeros(T, horizon_total);
    Div_growth = zeros(T, horizon_total);

    t_growth = zeros(T, horizon_total);
    g_growth = zeros(T, horizon_total);
    div_growth = zeros(T, horizon_total);

    coeff_T_sim1 = zeros(1, horizon_total);
    coeff_G_sim1 = zeros(1, horizon_total);
    coeff_Div_sim1 = zeros(1, horizon_total);

    for horizon = 1:horizon_total
        t_growth(1 + horizon:end, horizon) = log(t_ratio_sim(1 + horizon:end, 1)) - log(t_ratio_sim(1:end - horizon, 1));
        g_growth(1 + horizon:end, horizon) = log(g_ratio_sim(1 + horizon:end, 1)) - log(g_ratio_sim(1:end - horizon, 1));
        div_growth(1 + horizon:end, horizon) = log(div_ratio_sim(1 + horizon:end, 1)) - log(div_ratio_sim(1:end - horizon, 1));

        tmp = movsum(gdp_growth_sim, [horizon - 1 0]);
        GDP_growth(1 + horizon:end, horizon) = tmp(1 + horizon:end);

        T_growth(:, horizon) = GDP_growth(:, horizon) + t_growth(:, horizon);
        G_growth(:, horizon) = GDP_growth(:, horizon) + g_growth(:, horizon);
        Div_growth(:, horizon) = GDP_growth(:, horizon) + div_growth(:, horizon);
    end

    T = size(surplus, 1);

    for horizon = 1:50
        Y_Beta = T_growth(1 + horizon:end, horizon);
        X_Beta = GDP_growth(1 + horizon:end, horizon);
        mdl = LinearModel.fit([X_Beta(:, 1)], Y_Beta);
        tmp_se = mdl.Coefficients.SE(2);

        k = horizon;
        coeff_T_sim1(k) = mdl.Coefficients.Estimate(2);
    end

    coeff_T_sim(nboot, :) = coeff_T_sim1;

    for horizon = 1:50
        Y_Beta = G_growth(1 + horizon:end, horizon);
        X_Beta = GDP_growth(1 + horizon:end, horizon);
        mdl = LinearModel.fit([X_Beta(:, 1)], Y_Beta);
        tmp_se = mdl.Coefficients.SE(2);
        k = horizon;
        coeff_G_sim1(k) = mdl.Coefficients.Estimate(2);
    end

    coeff_G_sim(nboot, :) = coeff_G_sim1;

    for horizon = 1:50
        Y_Beta = Div_growth(1 + horizon:end, horizon);
        X_Beta = GDP_growth(1 + horizon:end, horizon);
        mdl = LinearModel.fit([X_Beta(:, 1)], Y_Beta);
        tmp_se = mdl.Coefficients.SE(2);
        k = horizon;
        coeff_Div_sim1(k) = mdl.Coefficients.Estimate(2);
    end

    coeff_Div_sim(nboot, :) = coeff_Div_sim1;
end

mean_coeff_T = nanmean(coeff_T_sim, 1);
mean_coeff_G = nanmean(coeff_G_sim, 1);
mean_coeff_Div = nanmean(coeff_Div_sim, 1);
std_coeff_T = nanstd(coeff_T_sim, 1);
std_coeff_G = nanstd(coeff_G_sim, 1);
std_coeff_Div = nanstd(coeff_Div_sim, 1);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% plot
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

horizon = 10;

f = figure;
aux_plot_CI_beta(1:horizon, coeff_G(1, 1:horizon), std_coeff_G(1:horizon))
set(gca, 'FontName', 'Times New Roman')
set(gca, 'FontSize', 12);
f.PaperSize = [6 4];
filename = strcat('../figures/pic-beta_G', '.pdf')
print(filename, '-dpdf', '-fillpage');

f = figure;
aux_plot_CI_beta(1:horizon, coeff_T(1, 1:horizon), std_coeff_T(1:horizon))
set(gca, 'FontName', 'Times New Roman')
set(gca, 'FontSize', 12);
f.PaperSize = [6 4];
filename = strcat('../figures/pic-beta_T', '.pdf')
print(filename, '-dpdf', '-fillpage');
