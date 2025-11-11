clear; close all;

load("MAT/result_step3.mat");
ttime = [1947:2020];
loss = 0.40;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% fig. Peso
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
I_gstrip = (I_dg + I_gdp + I_pi);
Ag(1) = -y0nom_1 + g0 + x0 + pi0 + .5 * (I_gstrip)' * (Sig * Sig') * (I_gstrip) - (I_gstrip)' * Sig * L0;
Bg(:, 1) = ((I_gstrip)' * Psi - I_y1' - (I_gstrip)' * Sig * L1)';
PDg_model = exp(Ag(1) + Bg(:, 1)' * X2t')';

for j = 1:(striphorizon - 1)
    Ag(j + 1) = Ag(j) -y0nom_1 + g0 + x0 + pi0 + .5 * (I_gstrip + Bg(:, j))' * (Sig * Sig') * (I_gstrip + Bg(:, j)) - (I_gstrip + Bg(:, j))' * Sig * L0;
    Bg(:, j + 1) = ((I_gstrip + Bg(:, j))' * Psi - I_y1' - (I_gstrip + Bg(:, j))' * Sig * L1)';
    PDg_model = PDg_model + exp(Ag(j + 1) + Bg(:, j + 1)' * X2t')';
end

PDg_model_breakdown(:, 1) = PDg_model;

for j = 2:striphorizon
    PDg_model_breakdown(:, j) = PDg_model_breakdown(:, j - 1) - exp(Ag(j - 1) + Bg(:, j - 1)' * X2t')';
end

I_tstrip = (I_dt + I_gdp + I_pi);
At(1) = -y0nom_1 + tau0 + x0 + pi0 + .5 * (I_tstrip)' * (Sig * Sig') * (I_tstrip) - (I_tstrip)' * Sig * L0;
Bt(:, 1) = ((I_tstrip)' * Psi - I_y1' - (I_tstrip)' * Sig * L1)';
PDt_model = exp(At(1) + Bt(:, 1)' * X2t')';

for j = 1:(striphorizon - 1)
    At(j + 1) = At(j) -y0nom_1 + tau0 + x0 + pi0 + .5 * (I_tstrip + Bt(:, j))' * Sig * Sig' * (I_tstrip + Bt(:, j)) - (I_tstrip + Bt(:, j))' * Sig * L0;
    Bt(:, j + 1) = ((I_tstrip + Bt(:, j))' * Psi - I_y1' - (I_tstrip + Bt(:, j))' * Sig * L1)';
    PDt_model = PDt_model + exp(At(j + 1) + Bt(:, j + 1)' * X2t')';
end

SGDP = (1 + PDt_model) .* taxrevgdp - (1 + PDg_model) .* spendgdp;

GapGDP = gdebt(2:end)' - SGDP;
dd = 1:striphorizon;

parfor i = 1:T
    f = @(phi) (sum((1 - phi) .^ (dd - 1) .* phi .* PDg_model_breakdown(i, dd)) * spendgdp(i) * loss - GapGDP(i));
    [phi(i), val(i), flag(i)] = fsolve(f, 0.01);
end

sum(flag <= 0)
mean(phi)

prod(1 - phi)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Countercyclical Peso Probability
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
nu = 0.02;

opts = optimoptions('fsolve', 'MaxFunctionEvaluations', 1e4);

parfor i = 1:T
    f = @(phi) (loss * (PDg_model(i) - helper_peso_sumV(phi, X2t(i, :), y0nom_1, g0, x0, pi0, nu, Sigma, Sig, Psi, L0, L1, ...
        I_y1, I_gdp, I_gstrip, striphorizon)) * spendgdp(i) - GapGDP(i));
    [phinew(i), val(i), flag(i)] = fsolve(f, 0.1, opts);

    fraction_of_G(i) = loss * helper_peso_sumV(phinew(i), X2t(i, :), y0nom_1, g0, x0, pi0, nu, Sigma, Sig, Psi, L0, L1, ...
        I_y1, I_gdp, I_gstrip, striphorizon) / PDg_model(i);
end

sum(flag <= 0)
mean(phinew)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% plot
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

f = figure;
plot(ttime, phi * 100, ttime, phinew * 100, 'LineWidth', 3)
xlim([1947 2020])
set(gca, 'FontName', 'Times New Roman')
set(gca, 'FontSize', 12);
grid
legend('Baseline Peso', 'Countercyclical Peso')
ylabel('Implied Spending Cut Probability (%)')
f.PaperSize = [6 4];
print('../figures/pic-peso', '-dpdf', '-fillpage');

phicut = mean(phi) * 100;
phicut_new = mean(phinew) * 100;
twosd_gdpshock = Sig(gdppos, gdppos) * 2 * 100;
twosd_cutprob = (exp(nu * 2) - 1) * 100;
save('MAT/peso.mat', 'nu', 'phi', 'phinew', 'phicut', 'phicut_new', ...
    'loss', 'twosd_gdpshock', 'twosd_cutprob');

% risk premium
phi = phinew(end);
X2t = X2t(end, :);
Apeso(1) = log(1 - phi) -y0nom_1 + g0 + x0 + pi0 + ...
    1/2 * (I_gstrip)' * Sigma * (I_gstrip) - (I_gstrip)' * Sig * L0 + ...
    (-L0' + (I_gstrip)' * Sig) * (nu * I_gdp);

Bpeso(:, 1) = ((I_gstrip)' * Psi - I_y1' ...
    - (I_gstrip' * Sig + nu * I_gdp') * L1)';

PDpeso_model = exp(Apeso(1) + Bpeso(:, 1)' * X2t')';

for j = 1:(striphorizon - 1)
    Apeso(j + 1) = log(1 - phi) + Apeso(j) -y0nom_1 + g0 + x0 + pi0 + ...
        1/2 * (I_gstrip + Bpeso(:, j))' * Sigma * (I_gstrip + Bpeso(:, j)) - (I_gstrip + Bpeso(:, j))' * Sig * L0 + ...
        (-L0' + (I_gstrip + Bpeso(:, j))' * Sig) * (nu * I_gdp);

    Bpeso(:, j + 1) = ((I_gstrip + Bpeso(:, j))' * Psi - I_y1' ...
        - ((I_gstrip + Bpeso(:, j))' * Sig + nu * I_gdp') * L1)';

    PDpeso_model = PDpeso_model + exp(Apeso(j + 1) + Bpeso(:, j + 1)' * X2t')';
end

horizon = 50;

f = figure;
plot(1:horizon, (-Ag(1:horizon) ./ (1:horizon) - y0nom_1 + (g0 + x0 + pi0)) * 100, 'b', 'LineWidth', 3)
hold on
plot(1:horizon, (-Apeso(1:horizon) ./ (1:horizon) + log(1 - phi) - y0nom_1 + (g0 + x0 + pi0)) * 100, 'k', 'LineWidth', 3)
plot(1:horizon, (-At(1:horizon) ./ (1:horizon) - y0nom_1 + (tau0 + x0 + pi0)) * 100, 'r--', 'LineWidth', 3)
plot(1:horizon, (-Ax(1:horizon) ./ (1:horizon) - y0nom_1 + (x0 + pi0)) * 100, 'g', 'LineStyle', '-.', 'LineWidth', 3)

hold off
set(gca, 'FontName', 'Times New Roman')
set(gca, 'FontSize', 12);
legend('G Claim', 'G Cut Claim', 'T Claim', 'GDP', 'Location', 'Southeast')
xlabel('Period (Year)')
ylabel('Risk Premium (%)')
grid
f.PaperSize = [6 4];
print('../figures/peso_cum_rp', '-dpdf', '-fillpage');
