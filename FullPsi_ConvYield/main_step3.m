clear all; close all; clc

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Load data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
LoadData_ConvYield;
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
    param = zeros(42, 1);
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
        [param, fval, exitflag, output] = fminsearch('solve_step3', param_init, options, N, T, Psi, Sig, I_pi, I_gdp, I_y1, I_yspr, I_pdm, I_divgrm, I_dg, divgrpos, ...
            mu_m, k1m, r0_m, A0m, striphorizon, y0nom_1, y0nom_5, pi0, x0, g0, X2t, tau, yielddata, tipstau, tipsdata, dvd, lev, eps2);
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

L1(I_pi == 1, I_pi == 1) = param(i); i = i + 1;
L1(I_pi == 1, I_y1 == 1) = param(i); i = i + 1;
L1(I_pi == 1, I_yspr == 1) = param(i); i = i + 1;
L1(I_pi == 1, I_gdp == 1) = param(i); i = i + 1;
L1(I_yspr == 1, 1:N) = param(i:i + N - 1); i = i + N;
L1(I_gdp == 1, 1:N) = param(i:i + N - 1); i = i + N;
L1(I_pdm == 1, 1:N) = param(i:i + N - 1); i = i + N;

Model_Summary;

save('MAT/result_step3.mat');

pvs_withcy = ((PDt_model) .* taxrevgdp - (PDg_model) .* spendgdp);
horizon = 100;
rpt_withcy = (-At(1:horizon) ./ (1:horizon) - y0nom_1 + (tau0 + x0 + pi0)) * 100;
rpg_withcy = (-Ag(1:horizon) ./ (1:horizon) - y0nom_1 + (tau0 + x0 + pi0)) * 100;
save('MAT/result_pv.mat', 'pvs_withcy', 'rpt_withcy', 'rpg_withcy');
