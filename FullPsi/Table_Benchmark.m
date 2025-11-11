clear; close all;

load('MAT/result_step3.mat');

%% Lambda Table
fID = fopen(['../tables/TableLambda.tex'], 'w');

L_0_char = ['a'; 'b'; 'c'; 'd'; 'e'];
l0namebegin = '\\newcommand{\\WithoutDebtLamC';
l0nameend1 = '}{%4.2f}\n';
l0nameend2 = '}{%4.0f}\n';

L0_val = [L0(1); L0(3:5); L0(7)];
% L0 handles
for i = 1:5
    fprintf(fID, [l0namebegin L_0_char(i) l0nameend1], L0_val(i));
end

% L1 handles
L_1_char = ['a'; 'b'; 'c'; 'd'; 'e'; 'f'; 'g'; 'h'; 'i'; 'j'; 'k'];

l1namebegin = '\\newcommand{\\WithoutDebtLamT';
l1nameend1 = '}{%4.2f}\n';
l1nameend2 = '}{%4.0f}\n';

fprintf(fID, '\\newcommand{\\WithoutDebtLamTa}{%4.2f}\n', L1(1, 1));
fprintf(fID, '\\newcommand{\\WithoutDebtLamTb}{%4.2f}\n', L1(1, 2));
fprintf(fID, '\\newcommand{\\WithoutDebtLamTc}{%4.2f}\n', L1(1, 3));
fprintf(fID, '\\newcommand{\\WithoutDebtLamTd}{%4.2f}\n', L1(1, 4));

% L1 handles row 3
for i = 1:N
    fprintf(fID, [l1namebegin 'c' L_1_char(i) l1nameend1], L1(3, i));
end

% L1 handles row 4
for i = 1:N
    fprintf(fID, [l1namebegin 'd' L_1_char(i) l1nameend1], L1(4, i));
end

% L1 handles row 7
for i = 1:N
    fprintf(fID, [l1namebegin 'g' L_1_char(i) l1nameend1], L1(7, i));
end

fclose(fID);

%%
fID = fopen(['../tables/TableRates.tex'], 'w'); 

fprintf(fID, '\\newcommand{\\WithoutDebtFiveyrnomavg}{%4.2f}\n', 100 * mean(ynom5_model));
fprintf(fID, '\\newcommand{\\WithoutDebtFiveyrrealavg}{%4.2f}\n', 100 * mean(yreal5_model));
fprintf(fID, '\\newcommand{\\WithoutDebtFiveyrexpinfavg}{%4.2f}\n', 100 * mean(expinfl5));
fprintf(fID, '\\newcommand{\\WithoutDebtFiveyrinflrpavg}{%4.2f}\n', 100 * mean(inflrp5_model));

fprintf(fID, '\\newcommand{\\WithoutDebtmutau}{%4.2f}\n', 100 * mean(deltalogtau));
fprintf(fID, '\\newcommand{\\WithoutDebtmug}{%4.2f}\n', 100 * mean(deltalogg));
fprintf(fID, '\\newcommand{\\WithoutDebtmeangdpnom}{%4.2f}\n', 100 * (mean(rpcgdpgr) + mean(inflation)));
fprintf(fID, '\\newcommand{\\WithoutDebtmeanynom}{%4.2f}\n', 100 * (mean(ynom1)));

% Decomposing the risk premium on T and G claims
fprintf(fID, '\\newcommand{\\WithoutDebtPDTannmean}{%4.2f}\n', mean((PDt_model)));
fprintf(fID, '\\newcommand{\\WithoutDebtrpTannmean}{%4.2f}\n', expret_t2 * 100);
fprintf(fID, '\\newcommand{\\WithoutDebtexpretT}{%4.2f}\n', expret_t3 * 100);

fprintf(fID, '\\newcommand{\\WithoutDebtrpTannmeans}{%4.2f}\n', mean((I_gdp + I_pi + I_dt + Bt(:, 1:3))' * Sig * L0 * 100));
fprintf(fID, '\\newcommand{\\WithoutDebtrtunconditional}{%4.2f}\n', r0t * 100);
fprintf(fID, '\\newcommand{\\WithoutDebtrtrealunconditional}{%4.2f}\n', (r0t - pi0) * 100);

contret_tclaim = aaa_t .* rpp;
fprintf(fID, '\\newcommand{\\WithoutDebttclaiminflret}{%4.2f}\n', contret_tclaim(1));
fprintf(fID, '\\newcommand{\\WithoutDebttclaimgdpret}{%4.2f}\n', contret_tclaim(2));
fprintf(fID, '\\newcommand{\\WithoutDebttclaimrfret}{%4.2f}\n', contret_tclaim(3));
fprintf(fID, '\\newcommand{\\WithoutDebttclaimyspdret}{%4.2f}\n', contret_tclaim(4));
fprintf(fID, '\\newcommand{\\WithoutDebttclaimdgret}{%4.2f}\n', contret_tclaim(5));

fprintf(fID, '\\newcommand{\\WithoutDebtGDPrp}{%4.2f}\n', gdp_rp_minus_longbond_model_wJensen * 100);
fprintf(fID, '\\newcommand{\\WithoutDebtrpmannmean}{%4.2f}\n', mean(equityriskpremTV_model));
fprintf(fID, '\\newcommand{\\WithoutDebtPDmannmean}{%4.2f}\n', mean(exp(pdm)));

fprintf(fID, '\\newcommand{\\WithoutDebtPDGannmean}{%4.2f}\n', mean((PDg_model)));
fprintf(fID, '\\newcommand{\\WithoutDebtrpGannmean}{%4.2f}\n', expret_g2 * 100);
fprintf(fID, '\\newcommand{\\WithoutDebtexpretG}{%4.2f}\n', expret_g3 * 100);

fprintf(fID, '\\newcommand{\\WithoutDebtrpGannmeans}{%4.2f}\n', mean((I_gdp + I_pi + I_dg + Bg(:, 1:3))' * Sig * L0 * 100));
fprintf(fID, '\\newcommand{\\WithoutDebtrgunconditional}{%4.2f}\n', r0g * 100);

fprintf(fID, '\\newcommand{\\WithoutDebtgordong}{%4.2f}\n', 1 / (r0g - (mean(rpcgdpgr) + mean(inflation))));
fprintf(fID, '\\newcommand{\\WithoutDebtgordont}{%4.2f}\n', 1 / (r0t - (mean(rpcgdpgr) + mean(inflation))));

contret_gclaim = aaa_g .* rpp;
fprintf(fID, '\\newcommand{\\WithoutDebtgclaiminflret}{%4.2f}\n', contret_gclaim(1));
fprintf(fID, '\\newcommand{\\WithoutDebtgclaimgdpret}{%4.2f}\n', contret_gclaim(2));
fprintf(fID, '\\newcommand{\\WithoutDebtgclaimrfret}{%4.2f}\n', contret_gclaim(3));
fprintf(fID, '\\newcommand{\\WithoutDebtgclaimyspdret}{%4.2f}\n', contret_gclaim(4));
fprintf(fID, '\\newcommand{\\WithoutDebtgclaimdgret}{%4.2f}\n', contret_gclaim(5));

fprintf(fID, '\\newcommand{\\WithoutDebtnomgdpg}{%4.2f}\n', (pi0 + x0) * 100);
fprintf(fID, '\\newcommand{\\WithoutDebtrealgdpg}{%4.2f}\n', x0 * 100);
fprintf(fID, '\\newcommand{\\WithoutDebtinfl}{%4.2f}\n', pi0 * 100);
fprintf(fID, '\\newcommand{\\WithoutDebtrxunconditional}{%4.2f}\n', r0x * 100);

maxmat = 1000;
fprintf(fID, '\\newcommand{\\WithoutDebtltnomyeild}{%4.2f}\n', -100 * Api(maxmat) ./ maxmat);
fprintf(fID, '\\newcommand{\\WithoutDebtltrealyeild}{%4.2f}\n', -100 * A(maxmat) ./ maxmat);

PDg_model0 = sum(exp(Ag));
PDt_model0 = sum(exp(At));
fprintf(fID, '\\newcommand{\\WithoutDebtmeantau}{%4.2f}\n', 100 * mean(taxrevgdp));
fprintf(fID, '\\newcommand{\\WithoutDebtmeang}{%4.2f}\n', 100 * mean(spendgdp));
fprintf(fID, '\\newcommand{\\WithoutDebtpdtauzero}{%4.2f}\n', (PDt_model0));
fprintf(fID, '\\newcommand{\\WithoutDebtpdgzero}{%4.2f}\n', (PDg_model0));
fprintf(fID, '\\newcommand{\\WithoutDebtpvszero}{%4.2f}\n', ...
    (PDt_model0) * 100 * mean(taxrevgdp) - (PDg_model0) * 100 * mean(spendgdp));

TGDP = (PDt_model) .* taxrevgdp;
GGDP = (PDg_model) .* spendgdp;
SGDP = (PDt_model) .* taxrevgdp - (PDg_model) .* spendgdp;

TPGDP = (PDt_model) .* taxrevgdp ./ (1 + PDx_model);
GPGDP = (PDg_model) .* spendgdp ./ (1 + PDx_model);
SPGDP = ((PDt_model) .* taxrevgdp - (PDg_model) .* spendgdp) ./ (1 + PDx_model);

fprintf(fID, '\\newcommand{\\WithoutDebtTGDP}{%4.2f}\n', mean(TGDP) * 1);
fprintf(fID, '\\newcommand{\\WithoutDebtGGDP}{%4.2f}\n', mean(GGDP) * 1);
fprintf(fID, '\\newcommand{\\WithoutDebtSGDP}{%4.2f}\n', mean(SGDP) * 100);

fprintf(fID, '\\newcommand{\\WithoutDebtTPGDP}{%4.2f}\n', mean(TPGDP) * 100);
fprintf(fID, '\\newcommand{\\WithoutDebtGPGDP}{%4.2f}\n', mean(GPGDP) * 100);
fprintf(fID, '\\newcommand{\\WithoutDebtSPGDP}{%4.2f}\n', mean(SPGDP) * 100);

fprintf(fID, '\\newcommand{\\WithoutDebtDGDP}{%4.2f}\n', mean(gdebt) * 100);

fprintf(fID, '\\newcommand{\\WithoutDebtGapGDP}{%4.0f}\n', ...
    (mean(gdebt) - mean(SGDP)) * 100);
fprintf(fID, '\\newcommand{\\WithoutDebtDGDPlastTwenty}{%4.2f}\n', mean(gdebt(end - 20:end)) * 100);
fprintf(fID, '\\newcommand{\\WithoutDebtDGDPlastTen}{%4.0f}\n', mean(gdebt(end - 10:end)) * 100);
fprintf(fID, '\\newcommand{\\WithoutDebtSGDPlastTwenty}{%4.0f}\n', (-1) * mean(SGDP(end - 20:end)) * 100);
fprintf(fID, '\\newcommand{\\WithoutDebtSGDPlastTen}{%4.0f}\n', (-1) * mean(SGDP(end - 10:end)) * 100);

fprintf(fID, '\\newcommand{\\WithoutDebtGapGDPlastOne}{%4.0f}\n', ...
    (gdebt(end) - SGDP(end)) * 100);

load('../FullPsi_ConvYield/MAT/result_pv.mat', 'pvs_withcy');
fprintf(fID, '\\newcommand{\\WithCYGapGDP}{%4.2f}\n', ...
    (mean(pvs_withcy)) * 100);

% risk premium
horizon = 1:50;
rp_gstrip = -Ag(1:horizon) ./ (1:horizon) - y0nom_1 + (g0 + x0 + pi0);
rp_tstrip = -At(1:horizon) ./ (1:horizon) - y0nom_1 + (tau0 + x0 + pi0);
fprintf(fID, '\\newcommand{\\WithoutDebtRPToneyear}{%0.2f}\n', rp_tstrip(1) * 100);
fprintf(fID, '\\newcommand{\\WithoutDebtRPGoneyear}{%0.2f}\n', rp_gstrip(1) * 100);

% cash flow
fprintf(fID, '\\newcommand{\\WithoutDebtmeangdata}{%4.2f}\n', (mean(spendgdp) * 100));
fprintf(fID, '\\newcommand{\\WithoutDebtmeantdata}{%4.2f}\n', (mean(taxrevgdp) * 100));

% debt and surplus value vol
fprintf(fID, '\\newcommand{\\WithoutDebtdebtgdpvol}{%4.2f}\n', std(gdebt));
fprintf(fID, '\\newcommand{\\WithoutDebtddebtgdpvol}{%4.2f}\n', std(gdebt(2:end) - gdebt(1:end - 1)) * 100);
fprintf(fID, '\\newcommand{\\WithoutDebtpvsurplusvol}{%4.2f}\n', std(SGDP));
fprintf(fID, '\\newcommand{\\WithoutDebtdpvsurplusvol}{%4.2f}\n', std(SGDP(2:end) - SGDP(1:end - 1)) * 100);

% peso cut
load('MAT/peso.mat');

fprintf(fID, '\\newcommand{\\WithoutDebtpesocutg}{%4.2f}\n', 100 * mean(spendgdp) * (1 - loss));
fprintf(fID, '\\newcommand{\\pesonu}{%4.0f}\n', 100 * nu);
fprintf(fID, '\\newcommand{\\phicut}{%4.2f}\n', phicut);
fprintf(fID, '\\newcommand{\\phicutnew}{%4.2f}\n', phicut_new);
fprintf(fID, '\\newcommand{\\twosdgdpshock}{%4.2f}\n', twosd_gdpshock);
fprintf(fID, '\\newcommand{\\twosdcutprob}{%4.0f}\n', twosd_cutprob);
fprintf(fID, '\\newcommand{\\pesochainedprob}{%4.2f}\n', prod(1 - phi) * 100);
fprintf(fID, '\\newcommand{\\pesochainedprobnew}{%4.2f}\n', prod(1 - phinew) * 100);

load('../RawData/duration_aggr_46_20.mat');
date = round(datem / 100);
date = date(min(find(date == 1947)):end, :);
datey = unique(date);

for i = 1:length(datey)
    durationy (i) = mean(duration.aggr(find(date == datey(i)))) / 365;
end

fprintf(fID, '\\newcommand{\\durationmean}{%4.2f}\n', nanmean(durationy));

totalret = xlsread('../RawData/MaindatafileA_Feb2021_longsample.xlsx', 'VAR2', 'AK20:AK93');
threemoncmt = xlsread('../RawData/MaindatafileA_Feb2021_longsample.xlsx', 'VAR2', 'h20:h93');

fprintf(fID, '\\newcommand{\\exretnave}{%4.2f}\n', nanmean(totalret - threemoncmt / 100) * 100);

fclose(fID);
