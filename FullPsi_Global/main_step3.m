clear all; close all; 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Load data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
LoadData_Global;
load_dividendstrips;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Estimation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
option.estimate.initial = 1; % 0. initial estimate; 1. iterate from the previous estimation
option.estimate.iter = 0; % loop in each estimation step: 0 means showing estimation fit
striphorizon = 4000;

options = optimset('DiffMinChange', 1e-2, ...
    'TolX', 1e-6, 'TolFun', 1e-6, 'MaxIter', 1e5, 'MaxFunEval', 1e8, 'Display', 'iter');

load(['MAT/SDF/x_step1.mat'], 'param');
param0 = param;
load(['MAT/SDF/x_step2.mat'], 'param');
param1 = param;

filename = ['MAT/SDF/x_step3.mat'];

if option.estimate.initial == 0
    param = zeros(68, 1);
    param(1:3) = param0(1:3);
    param(4:end) = param1;
elseif option.estimate.initial == 1
    load(filename, 'param');
end

if option.estimate.iter > 0

    for i = 1:option.estimate.iter
        param_init = param;
        global counts
        counts = 0;
        [param, fval, exitflag, output] = fminsearch('solve_step3', param_init, options, N, T, Psi, Sig, I_pi, I_gdp, I_y1, I_yspr, I_pdm, I_divgrm, I_dg, divgrpos, I_pi_g, I_gdp_g, I_equityreg, ...
            mu_m, k1m, r0_m, A0m, striphorizon, y0nom_1, y0nom_5, pi0, x0, g0, X2t, tau, yielddata, tipstau, tipsdata, dvd, lev, eret0, eps2);
        res_fval(i) = fval;
        save(filename, 'param', 'res_fval');
    end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Fill in Lambda
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
L0 = zeros(N, 1);
L1 = zeros(N, N);

i = 1;

L0(I_pi == 1) = param(i); i = i + 1;
L0(I_yspr == 1) = param(i); i = i + 1;
L0(I_gdp == 1) = param(i); i = i + 1;
L0(I_divgrm == 1) = param(i); i = i + 1;
L0(I_pdm == 1) = param(i); i = i + 1;

L0(I_pi_g == 1) = param(i); i = i + 1;
L0(I_gdp_g == 1) = param(i); i = i + 1;
L0(I_equityreg == 1) = param(i); i = i + 1;

L1(I_pi == 1, I_pi == 1) = param(i); i = i + 1;
L1(I_pi == 1, I_y1 == 1) = param(i); i = i + 1;
L1(I_pi == 1, I_yspr == 1) = param(i); i = i + 1;
L1(I_pi == 1, I_gdp == 1) = param(i); i = i + 1;
L1(I_yspr == 1, 1:N) = param(i:i + N - 1); i = i + N;
L1(I_gdp == 1, 1:N) = param(i:i + N - 1); i = i + N;
L1(I_pdm == 1, 1:N) = param(i:i + N - 1); i = i + N;
L1(I_equityreg == 1, 1:N) = param(i:i + N - 1); i = i + N;


Model_Summary;

save('MAT/result_step3.mat');

step = 3;

ttime = [1947:2020];

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
print(['../figures/pic-global-puzzle-step', num2str(step)], '-dpdf', '-fillpage');
