%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Load data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
mdata = xlsread('../RawData/MaindatafileA_Feb2021_longsample.xlsx', 'VAR2', 'a20:aj93'); % 1947-2020
mdata0 = xlsread('../RawData/MaindatafileA_Feb2021_longsample.xlsx', 'VAR2', 'a19:aj19'); % 1946

option.figure = 0;

Tstart = 1; % start sample in 1947
Tend = 74; % end sample in 2017
T = Tend - Tstart + 1;

date = mdata(Tstart:Tend, 1); % 1947-2020
taxrevgdp = mdata(Tstart:Tend, 2); % govmt tax revenue to GDP ratio
taxrevgdp0 = mdata0(1, 2); % govmt tax revenue to GDP ratio
spendgdp = mdata(Tstart:Tend, 3); % government spending before interest exp. to GDP ratio
spendgdp0 = mdata0(1, 3); % government spending before interest exp. to GDP ratio
surplusgdp = mdata(Tstart:Tend, 4); % primary surplus to gdp
rpcgdpgr = mdata(Tstart:Tend, 5); % real gdp growth (in logs)
inflation = mdata(Tstart:Tend, 6); % growth in GDP price deflator (in logs)
ynom0 = mdata(Tstart:Tend, 8) / 100; % nominal yield on a 3-month Treasury bill, rate from the last quarter of each year
ynom1 = mdata(Tstart:Tend, 9) / 100; % nominal yield on a 1-year Treasury bill, expressed per annum
ynom2 = mdata(Tstart:Tend, 10) / 100; % nominal yield on a 2-year Treasury note, expressed per annum
ynom5 = mdata(Tstart:Tend, 11) / 100; % nominal yield on a 5-year Treasury note, expressed per annum
ynom10 = mdata(Tstart:Tend, 12) / 100; % nominal yield on a 10-year Treasury note, expressed per annum
ynom20 = mdata(Tstart:Tend, 20) / 100; % nominal yield on a 20-year Treasury note, expressed per annum
ynom30 = mdata(Tstart:Tend, 13) / 100; % nominal yield on a 30-year Treasury note, expressed per annum
pdm = mdata(Tstart:Tend, 14); % log price-dividend ratio on CRSP vw stock index; dividend is annual and seasonally adjusted
divgrm = mdata(Tstart:Tend, 15); % nominal log dividend/GDP growth; dividend is annual and seasonally adjusted

ynom0 = log(1 + ynom0);
ynom1 = log(1 + ynom1);
ynom2 = log(1 + ynom2);
ynom5 = log(1 + ynom5);
ynom10 = log(1 + ynom10);
ynom20 = log(1 + ynom20);
ynom30 = log(1 + ynom30);

deltalogg = [log(spendgdp(1)) - log(spendgdp0); log(spendgdp(2:end)) - log(spendgdp(1:end - 1))];
deltalogtau = [log(taxrevgdp(1)) - log(taxrevgdp0); log(taxrevgdp(2:end)) - log(taxrevgdp(1:end - 1))];

yieldspr = ynom5 - ynom1;
y0nom_1 = mean(ynom1);
y0nom_5 = mean(ynom5);
yspr0 = mean(yieldspr);
pi0 = mean(inflation);
x0 = mean(rpcgdpgr);
tau0 = mean(deltalogtau);
g0 = mean(deltalogg);
surplus0 = mean(surplusgdp);

% log pd ratio on stocks
divgrm = divgrm - inflation - rpcgdpgr;

A0m = nanmean(pdm);
k1m = exp(A0m) / (exp(A0m) + 1);
k0m = log(exp(A0m) + 1) - k1m * A0m;
mu_m = nanmean(divgrm);
r0_m = mu_m + x0 + (log(exp(A0m) + 1) - A0m); % exp return consistent with present-value model

% select yield maturities for the term structure model
tau = [1, 2, 5, 10, 20, 30];
yielddata = [ynom1, ynom2, ynom5, ynom10, ynom20, ynom30];

tipstau = [5, 7, 10, 20, 30];
tipsdata = mdata(Tstart:Tend, 32:36) / 100;
tipsdata = log(1 + tipsdata);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Don't detrend tax and spending
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tts = taxrevgdp;
gts = spendgdp;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Read in debt/gdp
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
gdebt = xlsread('../RawData/nominal_aggr_debt_1946_2020_annual.xlsx', 'Sheet1', 'd2:d76')';

logdts = log(gdebt(2:end));
deltalogd = [logdts(1) - log(gdebt(1)), logdts(2:end) - logdts(1:(end - 1))];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Assembly
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% define the ordering in the VAR
inflpos = 1;
y1pos = 2;
ysprpos = 3;
gdppos = 4;
divgrpos = 5;
coint_div = 6;
pdpos = 7;
dtpos = 8;
coint_tax = 9;
dgpos = 10;
coint_spending = 11;

% VAR elements
infl = inflation - pi0;
x = rpcgdpgr - x0;
yield1 = ynom1 - y0nom_1;
nomyspr = yieldspr - yspr0;
dg = deltalogg - g0;
dt = deltalogtau - tau0;
pd = pdm - A0m;

X2(:, inflpos) = infl;
X2(:, y1pos) = yield1;
X2(:, ysprpos) = nomyspr;
X2(:, gdppos) = x;
X2(:, divgrpos) = divgrm - mean(divgrm);
X2(:, coint_div) = cumsum(divgrm) - mean(cumsum(divgrm));
X2(:, pdpos) = pd;
X2(:, dtpos) = dt;
X2(:, coint_tax) = log(tts) - mean(log(tts));
X2(:, dgpos) = dg;
X2(:, coint_spending) = log(gts) - mean(log(gts));

N = cols(X2);
I = eye(N);
I_pi = I(:, inflpos);
I_gdp = I(:, gdppos);
I_y1 = I(:, y1pos);
I_yspr = I(:, ysprpos);
I_pdm = I(:, pdpos);
I_divgrm = I(:, divgrpos);
I_dt = I(:, dtpos);
I_dg = I(:, dgpos);
I_coint_tax = I(:, coint_tax);
I_coint_spending = I(:, coint_spending);
I_coint_div = I(:, coint_div);

z_var = X2(2:end, :);
z_varlag = X2(1:end - 1, :);
Y = z_var;
X = z_varlag;

% actual, non-detrended series of tax and spending
X2t = X2;
X2t(:, coint_tax) = log(taxrevgdp) - mean(log(taxrevgdp));
X2t(:, coint_spending) = log(spendgdp) - mean(log(spendgdp));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Estimate Psi and Sig
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Psi = zeros(N, N);
Psi_se = zeros(N, N);

for i = [inflpos y1pos ysprpos gdppos divgrpos pdpos dtpos dgpos]
    regr = ols(Y(:, i), [ones(T - 1, 1), X(:, :)]);
    c(i) = regr.beta(1);
    c_se(i) = regr.bstd(1);
    Psi(i, :) = regr.beta(2:end)';
    Psi_se(i, :) = regr.bstd(2:end)';
    R2(i) = regr.rsqr;
    R2bar(i) = regr.rbar;
    eps(:, i) = Y(:, i) - c(i) - X * Psi(i, :)';
    clear regr;
end

Psi(coint_tax, :) = Psi(dtpos, :);
Psi(coint_spending, :) = Psi(dgpos, :);
Psi(coint_div, :) = Psi(divgrpos, :);
Psi(coint_tax, coint_tax) = Psi(coint_tax, coint_tax) + 1;
Psi(coint_spending, coint_spending) = Psi(coint_spending, coint_spending) + 1;
Psi(coint_div, coint_div) = Psi(coint_div, coint_div) + 1;

Psi_se(coint_tax, :) = Psi_se(dtpos, :);
Psi_se(coint_spending, :) = Psi_se(dgpos, :);
Psi_se(coint_div, :) = Psi_se(divgrpos, :);

Sigma = cov(eps(:, [inflpos y1pos ysprpos gdppos divgrpos pdpos dtpos dgpos]));
tmp = chol(Sigma, 'lower');
Sig = zeros(N, N);
Sig([inflpos y1pos ysprpos gdppos divgrpos], [inflpos y1pos ysprpos gdppos divgrpos]) = ...
    tmp(1:5, 1:5);
Sig(pdpos, [inflpos y1pos ysprpos gdppos divgrpos pdpos]) = tmp(6, 1:6);
Sig(dtpos, [inflpos y1pos ysprpos gdppos divgrpos pdpos dtpos]) = tmp(7, 1:7);
Sig(dgpos, [inflpos y1pos ysprpos gdppos divgrpos pdpos dtpos dgpos]) = tmp(8, 1:8);

Sig(coint_div, :) = Sig(divgrpos, :);
Sig(coint_tax, :) = Sig(dtpos, :);
Sig(coint_spending, :) = Sig(dgpos, :);

eps = [eps(:, [inflpos y1pos ysprpos gdppos divgrpos]), eps(:, divgrpos), ...
                eps(:, pdpos), eps(:, dtpos), eps(:, dtpos), ...
                eps(:, dgpos), eps(:, dgpos)];
Sigma = Sig * Sig';

tstat = Psi ./ Psi_se;

max(abs(eig(Psi)))

eps2 = eps;

tau0 = 0;
g0 = 0;
mu_m = 0;