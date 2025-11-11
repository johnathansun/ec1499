%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% For long sample, we use 10 year - 1 year yield as yield spread
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Load data
LoadData_LongSampleBreak;
ttime = [1930:2020];

%% Estimation
fn = @(x)([x(1) * (1 - x(2)) - x(3) - (x0 + pi0 - y0nom_1 - yspr0 - output_rp), ...
               x(2) - exp(x(1)) / (exp(x(1)) + 1), ...
               x(3) - (log(exp(x(1)) + 1) - x(2) * x(1))]);
options = optimoptions('fsolve', 'TolX', 1e-4, 'MaxFunctionEvaluations', 1e5, 'MaxIterations', 1e5);
[x, fval] = fsolve(fn, [0, 0, 0], options);

pxbar = x(1); %pdx
k1x = exp(pxbar) / (exp(pxbar) + 1); %rho
k0x = log(exp(pxbar) + 1) - k1x * pxbar; %;k0x

pmbar = A0m;
k1m = exp(pmbar) / (exp(pmbar) + 1);
k0m = log(exp(pmbar) + 1) - k1m * pmbar;
erp0 = x0 + pi0 - y0nom_1 - yspr0 - (pmbar * (1 - k1m) - k0m); % equity
erp = erp0 + (1 - k1m) * (- I_pdm' * X2t' + ((I_pi + I_gdp + I_divgrm)' * Psi - (I_y1 + I_yspr)') * inv(eye(N) - k1m * Psi) * X2t'); % equity

DR = (I_y1 + I_yspr)' * inv(eye(N) - k1x * Psi) * X2t';
DRM = (I_y1 + I_yspr)' * inv(eye(N) - k1m * Psi) * X2t' + (erp - erp0) / (1 - k1m);

CFX = (I_pi + I_gdp)' * Psi * inv(eye(N) - k1x * Psi) * X2t';
CFM = (I_pi + I_gdp + I_divgrm)' * Psi * inv(eye(N) - k1m * Psi) * X2t';
CFT = (I_pi + I_gdp + I_dt)' * Psi * inv(eye(N) - k1x * Psi) * X2t';
CFG = (I_pi + I_gdp + I_dg)' * Psi * inv(eye(N) - k1x * Psi) * X2t';

pdX = pxbar + CFX - DR;
pdT = pxbar + CFT - DR;
pdG = pxbar + CFG - DR;
pdM = pmbar + CFM - DRM;

s = exp(pdT) .* taxrevgdp' - exp(pdG) .* spendgdp';
upper = exp(pxbar) * (mean(taxrevgdp - spendgdp));

save(['MAT/CS_LongSample', num2str(output_rp * 1000), '.mat']);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% bootstrap
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
sc = parallel.pool.Constant(RandStream('Threefry'));
stream = sc.Value;
stream.Substream = 1;

Tlong = 1e5;
Xlong = zeros(Tlong, N);
shocks_indexlong = randsample(stream, 1:(T - 1), Tlong, true);

for i = 2:Tlong
    Xlong(i, :) = (Psi * Xlong(i - 1, :)')' + eps(shocks_indexlong(i), :);
end

nloop = 1e4;
results = zeros(nloop, T);

parfor iloop = 1:nloop
    stream = sc.Value;
    stream.Substream = iloop + 1;
    val = 1;

    while (val >= 1)
        init_index = randsample(stream, 1:Tlong, 1, true);
        shocks_index = randsample(stream, 1:(T - 1), T - 1, true);

        Xshort = zeros(T, N);
        Xshort(1, :) = Xlong(init_index, :);

        for i = 2:T
            Xshort(i, :) = (Psi * Xshort(i - 1, :)')' + eps(shocks_index(i - 1), :);
        end

        Psi1 = zeros(N, N);

        for i = 1:N
            regr = ols(Xshort(2:T, i), [ones(T - 1, 1), Xshort(1:T - 1, :)]);
            Psi1(i, :) = regr.beta(2:end)';
        end

        val = max(abs(eig(Psi1)));

        erp = erp0 + (1 - k1m) * (- I_pdm' * X2t' + ((I_pi + I_gdp + I_divgrm)' * Psi1 - (I_y1 + I_yspr)') * inv(eye(N) - k1m * Psi1) * X2t');
        DRM = (I_y1 + I_yspr)' * inv(eye(N) - k1m * Psi1) * X2t' + (erp - erp0) / (1 - k1m);
        CFM = (I_pi + I_gdp + I_divgrm)' * Psi1 * inv(eye(N) - k1m * Psi1) * X2t';
        pdM = pmbar + CFM - DRM;

        if (min(erp) < 0)
            val = 1;
        end

    end

    DR = (I_y1 + I_yspr)' * inv(eye(N) - k1x * Psi1) * X2t';

    CFX = (I_pi + I_gdp)' * Psi1 * inv(eye(N) - k1x * Psi1) * X2t';
    CFT = (I_pi + I_gdp + I_dt)' * Psi1 * inv(eye(N) - k1x * Psi1) * X2t';
    CFG = (I_pi + I_gdp + I_dg)' * Psi1 * inv(eye(N) - k1x * Psi1) * X2t';

    pdX = pxbar + CFX - DR;
    pdT = pxbar + CFT - DR;
    pdG = pxbar + CFG - DR;

    s1 = exp(pdT) .* taxrevgdp' - exp(pdG) .* spendgdp';

    results(iloop, :) = s1;
    resultt(iloop, :) = pdT;
    resultg(iloop, :) = pdG;
end

%% plot
f = figure;
std_coeff = std(results);
aux_plot_CI(ttime, s, std_coeff, gdebt, upper)
title(['GDP risk premium ', num2str(output_rp * 100), '%'])

axis([min(ttime) max(ttime) -3 1.5])

f.PaperSize = [6 4];
print(['../figures/cs_longsample_', num2str(output_rp * 1000)], '-dpdf', '-fillpage');
