%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Nominal and Real Rates, GDP Strips (x) and Dividend Strips (m)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
y0_1 = y0nom_1 - pi0 - .5 * I_pi' * (Sig * Sig') * I_pi + I_pi' * Sig * L0;
yts_1 = y0_1 + (I_y1' - I_pi' * Psi + I_pi' * Sig * L1) * X2t';

A = zeros(striphorizon, 1);
B = zeros(N, striphorizon);
Api = zeros(striphorizon, 1);
Bpi = zeros(N, striphorizon);
Am = zeros(striphorizon, 1);
Bm = zeros(N, striphorizon);

A(1) =- y0_1;
B(:, 1) =- (I_y1' - I_pi' * Psi + I_pi' * Sig * L1)';

Api(1) = -y0nom_1;
Bpi(:, 1) = -I_y1';

I_xstrip = (I_gdp + I_pi);
Ax(1) = -y0nom_1 + x0 + pi0 + .5 * (I_xstrip)' * (Sig * Sig') * (I_xstrip) - (I_xstrip)' * Sig * L0;
Bx(:, 1) = ((I_xstrip)' * Psi - I_y1' - (I_xstrip)' * Sig * L1)';
PDx_model = exp(Ax(1) + Bx(:, 1)' * X2t')';
PDx_model0 = exp(Ax(1))';

I_equitystrip = I_divgrm + I_gdp + I_pi;
Am(1) = mu_m + x0 - y0nom_1 + pi0 + .5 * (I_equitystrip)' * Sig * Sig' * (I_equitystrip) - (I_equitystrip)' * Sig * L0;
Bm(:, 1) = ((I_equitystrip)' * Psi - I_y1' - (I_equitystrip)' * Sig * L1)';
PDm_model = exp(Am(1) + Bm(:, 1)' * X2t')';
PDm_model0 = exp(Am(1))';

weightstrip = zeros(T, striphorizon);
weightstrip(:, 1) = exp(Am(1) + Bm(:, 1)' * X2t')';

varcapgain1 = zeros(N);
varcapgain2 = zeros(N);
adj_stock_j = zeros(1, T);

for j = 1:striphorizon - 1
    Api(j + 1) =- y0nom_1 + Api(j) + .5 * Bpi(:, j)' * (Sig * Sig') * Bpi(:, j) - Bpi(:, j)' * Sig * L0;
    Bpi(:, j + 1) = (Bpi(:, j)' * Psi - I_y1' - Bpi(:, j)' * Sig * L1)';

    A(j + 1) =- y0_1 + A(j) + .5 * B(:, j)' * (Sig * Sig') * B(:, j) - B(:, j)' * Sig * (L0 - Sig' * I_pi);
    B(:, j + 1) = ((I_pi + B(:, j))' * Psi - I_y1' - (I_pi + B(:, j))' * Sig * L1)';

    Ax(j + 1) = Ax(j) -y0nom_1 + x0 + pi0 + .5 * (I_xstrip + Bx(:, j))' * (Sig * Sig') * (I_xstrip + Bx(:, j)) - (I_xstrip + Bx(:, j))' * Sig * L0;
    Bx(:, j + 1) = ((I_xstrip + Bx(:, j))' * Psi - I_y1' - (I_xstrip + Bx(:, j))' * Sig * L1)';
    PDx_model = PDx_model + exp(Ax(j + 1) + Bx(:, j + 1)' * X2t')';
    PDx_model0 = PDx_model0 + exp(Ax(j + 1))';

    Am(j + 1) = Am(j) + mu_m + x0 - y0nom_1 + pi0 + .5 * (I_equitystrip + Bm(:, j))' * Sig * Sig' * (I_equitystrip + Bm(:, j)) - (I_equitystrip + Bm(:, j))' * Sig * (L0);
    Bm(:, j + 1) = ((I_equitystrip + Bm(:, j))' * Psi - I_y1' - (I_equitystrip + Bm(:, j))' * Sig * L1)';
    PDm_model = PDm_model + exp(Am(j + 1) + Bm(:, j + 1)' * X2t')';
    PDm_model0 = PDm_model0 + exp(Am(j + 1))';

    nombondriskprem(j) = 100 * Bpi(:, j)' * Sig * L0;
    realbondriskprem(j) = 100 * B(:, j)' * Sig * L0;
    equitydivstripriskprem(j) = 100 * (I_equitystrip + Bm(:, j))' * Sig * L0;
    gdpstripriskprem(j) = 100 * (I_gdp + I_pi + Bx(:, j))' * Sig * L0;

    nombondriskprem_woJensen(j) = 100 * (Bpi(:, j)' * Sig * L0 - .5 * Bpi(:, j)' * (Sig * Sig') * Bpi(:, j));
    realbondriskprem_woJensen(j) = 100 * (B(:, j)' * Sig * L0 - .5 * B(:, j)' * (Sig * Sig') * B(:, j));
    equitydivstripriskprem_woJensen(j) = 100 * ((I_equitystrip + Bm(:, j))' * Sig * L0 - .5 * (I_equitystrip +Bm(:, j))' * Sig * Sig' * (I_equitystrip + Bm(:, j)));
    gdpstripriskprem_woJensen(j) = 100 * ((I_gdp + I_pi + Bx(:, j))' * Sig * L0 - .5 * (I_pi + I_gdp +Bx(:, j))' * Sig * Sig' * (I_pi + I_gdp + Bx(:, j)));

    weightstrip(:, j + 1) = exp(Am(j + 1) + Bm(:, j + 1)' * X2t')';
end

pxbar = log(PDx_model0);
k1x = exp(pxbar) / (exp(pxbar) + 1);
Bxbar = inv(eye(N) - k1x * (Psi - Sig * L1)') * ((Psi - Sig * L1)' * I_xstrip - I_y1);
gdp_rp_model = (I_xstrip + Bxbar)' * Sig * L0;
gdp_rp_model_wJensen = (I_xstrip + Bxbar)' * Sig * L0 - ...
    .5 * (I_xstrip + Bxbar)' * (Sig * Sig') * (I_xstrip + Bxbar);
gdp_rp_minus_longbond_model_wJensen = gdp_rp_model_wJensen - yspr0;

Bmbar = inv(eye(N) - k1m * (Psi - Sig * L1)') * ((Psi - Sig * L1)' * I_equitystrip - I_y1);
equity_rp_model = (I_equitystrip + Bmbar)' * Sig * L0;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Spending (g), Tax (t) Strips
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

I_gstrip = (I_dg + I_gdp + I_pi);
Ag(1) = -y0nom_1 + g0 + x0 + pi0 + .5 * (I_gstrip)' * (Sig * Sig') * (I_gstrip) - (I_gstrip)' * Sig * L0;
Bg(:, 1) = ((I_gstrip)' * Psi - I_y1' - (I_gstrip)' * Sig * L1)';
PDg_model = exp(Ag(1) + Bg(:, 1)' * X2t')';
PDg_model0 = exp(Ag(1))';

for j = 1:(striphorizon - 1)
    Ag(j + 1) = Ag(j) -y0nom_1 + g0 + x0 + pi0 + .5 * (I_gstrip + Bg(:, j))' * (Sig * Sig') * (I_gstrip + Bg(:, j)) - (I_gstrip + Bg(:, j))' * Sig * L0;
    Bg(:, j + 1) = ((I_gstrip + Bg(:, j))' * Psi - I_y1' - (I_gstrip + Bg(:, j))' * Sig * L1)';
    PDg_model = PDg_model + exp(Ag(j + 1) + Bg(:, j + 1)' * X2t')';
    PDg_model0 = PDg_model0 + exp(Ag(j + 1))';
end

I_tstrip = (I_dt + I_gdp + I_pi);
At(1) = -y0nom_1 + tau0 + x0 + pi0 + .5 * (I_tstrip)' * (Sig * Sig') * (I_tstrip) - (I_tstrip)' * Sig * L0;
Bt(:, 1) = ((I_tstrip)' * Psi - I_y1' - (I_tstrip)' * Sig * L1)';
PDt_model = exp(At(1) + Bt(:, 1)' * X2t')';
PDt_model0 = exp(At(1))';

for j = 1:(striphorizon - 1)
    At(j + 1) = At(j) -y0nom_1 + tau0 + x0 + pi0 + .5 * (I_tstrip + Bt(:, j))' * Sig * Sig' * (I_tstrip + Bt(:, j)) - (I_tstrip + Bt(:, j))' * Sig * L0;
    Bt(:, j + 1) = ((I_tstrip + Bt(:, j))' * Psi - I_y1' - (I_tstrip + Bt(:, j))' * Sig * L1)';
    PDt_model = PDt_model + exp(At(j + 1) + Bt(:, j + 1)' * X2t')';
    PDt_model0 = PDt_model0 + exp(At(j + 1))';
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Evaluate Moments
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

aa = (I_equitystrip + k1m * I_pdm)' * Sig;
cc = ((I_equitystrip + k1m * I_pdm)' * Psi -I_pdm' -I_y1');
riskprem_m = r0_m + pi0 - y0nom_1 + 0.5 * aa * aa';
equityriskpremTV_model = 100 * (aa * L0 + aa * L1 * X2t');
equityriskpremTV_data = 100 * (riskprem_m + cc * X2t');
FF_vw = 100 * (riskprem_m - aa * L0); % pricing error is in percent per year

% Dynamics of 5-yr yield
ynom5_model = (-Api(5) / 5 - Bpi(:, 5)' / 5 * X2t')';
yreal5_model = (-A(5) / 5 - B(:, 5)' / 5 * X2t')';
% Expected inflation over the next 5 years, in quarterly terms
expinfl5 = (pi0 + (I_pi' * (I - Psi ^ 5) * inv(I - Psi) * X2t') / 5)';

% Nominal bond risk premia on 5-year bond
bondriskpremTV_model = (Api(1) - Api(5) / 5) + (-Bpi(:, 5)' / 5 + Bpi(:, 1)' / 5 * (I - Psi ^ 5) * inv(I - Psi)) * X2t';
bondriskpremTV_data = (y0nom_5 - y0nom_1) + (I_y1' + I_yspr' - I_y1' / 5 * (I - Psi ^ 5) * inv(I - Psi)) * X2t';

% Term structure of bond risk premia
for ttt = 1:400
    nombrp(ttt, :) = ((Api(1) - Api(ttt) / ttt) + (-Bpi(:, ttt)' / ttt + Bpi(:, 1)' / ttt * (I - Psi ^ ttt) * inv(I - Psi)) * X2t');
    realbrp(ttt, :) = ((A(1) - A(ttt) / ttt) + (-B(:, ttt)' / ttt + B(:, 1)' / ttt * (I - Psi ^ ttt) * inv(I - Psi)) * X2t');
    inflbrp(ttt, :) = nombrp(ttt, :) - realbrp(ttt, :);
    ts_brp_mean(ttt) = mean(nombrp(ttt, :));
    ts_brp_std(ttt) = std(nombrp(ttt, :));
    ts_brp_real_mean(ttt) = mean(realbrp(ttt, :));
    ts_brp_real_std(ttt) = std(realbrp(ttt, :));
    ts_brp_infl_mean(ttt) = mean(inflbrp(ttt, :));
    ts_brp_infl_std(ttt) = std(inflbrp(ttt, :));
end

% Inflation risk premium over 5-years (Ang-Bekaert or KvHVN definition), in quarterly terms
inflrp5_model = ynom5_model - yreal5_model - expinfl5;

divgrowth_stock = mu_m + X2t(:, divgrpos);
expdivgr_stock = mu_m + I_divgrm + I_gdp' * Psi * X2t';
logdiv_stock(1) = 160 + divgrowth_stock(1); % div normalized to 10 % of GDP in 1947

ptbar = log(mean(PDt_model));
k1t = exp(ptbar) / (exp(ptbar) + 1);
k0t = log(exp(ptbar) + 1) - k1t * ptbar;

pgbar = log(mean(PDg_model));
k1g = exp(pgbar) / (exp(pgbar) + 1);
k0g = log(exp(pgbar) + 1) - k1g * pgbar;

pxbar = log(mean(PDx_model));
k1x = exp(pxbar) / (exp(pxbar) + 1);
k0x = log(exp(pxbar) + 1) - k1x * pxbar;

Btbar = (eye(N) - k1t * (Psi - Sig * L1)') \ ((Psi - Sig * L1)' * (I_dt + I_gdp + I_pi) - I_y1);
Bgbar = (eye(N) - k1g * (Psi - Sig * L1)') \ ((Psi - Sig * L1)' * (I_dg + I_gdp + I_pi) - I_y1);
Bxbar = (eye(N) - k1x * (Psi - Sig * L1)') \ ((Psi - Sig * L1)' * (I_gdp + I_pi) - I_y1);

% Decomposing the risk premium on T and G claims
rpp = mean(Sig * (L0 + L1 * X2t') * 100, 2);
expret_g = (I_dg + I_gdp + I_pi + k1g * Bgbar)' * Sig * (L0) * 100;
expret_t = (I_dt + I_gdp + I_pi + k1t * Btbar)' * Sig * (L0) * 100;
aaa_g = (I_dg + I_gdp + I_pi + k1g * Bgbar);
aaa_t = (I_dt + I_gdp + I_pi + k1t * Btbar);

r0g = x0 +pi0 + k0g - pgbar * (1 - k1g);
r0t = x0 +pi0 + k0t - ptbar * (1 - k1t);
r0x = x0 +pi0 + k0x - pxbar * (1 - k1x);

expret_g2 = r0g - y0nom_1 + .5 * (I_dg + I_gdp + I_pi + k1g * Bgbar)' * Sig * Sig' * (I_dg + I_gdp + I_pi + k1g * Bgbar);
expret_t2 = r0t -y0nom_1 + .5 * (I_dt + I_gdp + I_pi + k1t * Btbar)' * Sig * Sig' * (I_dt + I_gdp + I_pi + k1t * Btbar);

% expected returns for G and T
expret_g3 = (I_dg + I_gdp + I_pi + k1g * Bgbar)' * Sig * (L0) + y0nom_1;
expret_t3 = (I_dt + I_gdp + I_pi + k1t * Btbar)' * Sig * (L0) + y0nom_1;
