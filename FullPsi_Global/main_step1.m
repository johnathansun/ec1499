clear all; close all; clc

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Load data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
LoadData_Global;
load_dividendstrips;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Estimation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
option.estimate.initial = 1; % 0. initial estimate; 1.iterate from the previous estimation
option.estimate.iter = 0; % loop in each estimation step: 0 means showing estimation fit
striphorizon = 4000;

options = optimset('DiffMinChange', 1e-2, ...
    'TolX', 1e-6, 'TolFun', 1e-6, 'MaxIter', 1e4, 'MaxFunEval', 1e8, 'Display', 'iter');

filename = ['MAT/SDF/x_step1.mat'];

if option.estimate.initial == 0
    param = zeros(1, 8);
elseif option.estimate.initial == 1
    load(filename, 'param');
end

if option.estimate.iter > 0

    for i = 1:option.estimate.iter
        param_init = param;
        global counts
        counts = 0;
        [param, fval, exitflag, output] = fminsearch('solve_step1', param_init, options, N, T, Psi, Sig, I_pi, I_gdp, I_y1, I_yspr, I_pdm, I_divgrm, divgrpos, I_pi_g, I_gdp_g, I_equityreg, ...
            mu_m, k1m, r0_m, A0m, striphorizon, y0nom_1, y0nom_5, pi0, x0, X2t, tau, yielddata, tipstau, tipsdata, dvd, lev, eret0, eps2);
        res_fval(i) = fval;
        save(filename, 'param');
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

Model_Summary;

save('MAT/result_step1.mat');
