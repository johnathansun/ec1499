clear; close all;

%% Saving parameter estimates to automatically update the paper
fID = fopen(['../tables/TablePsi.tex'], 'w');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Baseline Results
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
LoadData_Benchmark;

% Psi
Psichar = ['a'; 'b'; 'c'; 'd'; 'e'; 'f'; 'g'; 'h'; 'i'; 'j'; 'k'];
namebegin = '\\newcommand{\\WithoutDebtPPsi';
nameend1 = '}{%4.2f}\n';
nameend2 = '}{%4.0f}\n';

for ii = 1:N

    for jj = 1:N
        fprintf(fID, [namebegin Psichar(ii) Psichar(jj) nameend1], Psi(ii, jj));
    end

end

% maxeig and R2
fprintf(fID, '\\newcommand{\\maxeig}{%4.3f}\n', max(abs(eig(Psi))));
fprintf(fID, '\\newcommand{\\WithoutDebtRsqra}{%4.1f}\n', 100 * R2(1));
fprintf(fID, '\\newcommand{\\WithoutDebtRsqrb}{%4.1f}\n', 100 * R2(2));
fprintf(fID, '\\newcommand{\\WithoutDebtRsqrc}{%4.1f}\n', 100 * R2(3));
fprintf(fID, '\\newcommand{\\WithoutDebtRsqrd}{%4.1f}\n', 100 * R2(4));
fprintf(fID, '\\newcommand{\\WithoutDebtRsqre}{%4.1f}\n', 100 * R2(5));
fprintf(fID, '\\newcommand{\\WithoutDebtRsqrg}{%4.1f}\n', 100 * R2(7));
fprintf(fID, '\\newcommand{\\WithoutDebtRsqrh}{%4.1f}\n', 100 * R2(8));
fprintf(fID, '\\newcommand{\\WithoutDebtRsqrj}{%4.1f}\n', 100 * R2(10));

% T stat
Tstatchar = Psichar;
tnamebegin = '\\newcommand{\\WithoutDebtTstatt';
tnameend1 = '}{%4.2f}\n';
tnameend2 = '}{%4.0f}\n';

for ii = 1:N

    for jj = 1:N
        fprintf(fID, [tnamebegin Tstatchar(ii) Tstatchar(jj) tnameend1], tstat(ii, jj));
    end

end

% Sig
Sigchar = Psichar;
snamebegin = '\\newcommand{\\WithoutDebtSig';
snameend1 = '}{%4.2f}\n';
snameend2 = '}{%4.0f}\n';

for ii = 1:N

    for jj = 1:N
        fprintf(fID, [snamebegin Sigchar(ii) Sigchar(jj) snameend1], 100 * Sig(ii, jj));
    end

end

%% With Debt/GDP Results
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
LoadData_WithDebt;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% some summ stats
a1 = autocorr(log(gdebt(2:end)));
fprintf(fID, '\\newcommand{\\acfgdebt}{%4.3f}\n', a1(2));

Mdl = arima(1, 0, 0);
EstMdl = estimate(Mdl, log(gdebt(2:end))');
fprintf(fID, '\\newcommand{\\aronegdebt}{%4.3f}\n', 1.0078);

maskbreak1 = find(date <= 2007);
maskbreak2 = find(date > 2007);
logdts_break = log(gdebt(2:end));
logdts_break = [logdts_break(maskbreak1) - mean(logdts_break(maskbreak1)), ...
                    logdts_break(maskbreak2) - mean(logdts_break(maskbreak2))];

a1 = autocorr(logdts_break);
fprintf(fID, '\\newcommand{\\acfgdebtbreak}{%4.3f}\n', a1(2));

Mdl = arima(1, 0, 0);
EstMdl = estimate(Mdl, logdts_break');
fprintf(fID, '\\newcommand{\\aronegdebtbreak}{%4.3f}\n', EstMdl.AR{1});

ts = log(gdebt(2:end));

fprintf(fID, '\\newcommand{\\meangdebtpre}{%4.3f}\n', mean(ts(maskbreak1)));
fprintf(fID, '\\newcommand{\\meangdebtpost}{%4.3f}\n', mean(ts(maskbreak2)));
fprintf(fID, '\\newcommand{\\meangdebtdiff}{%4.3f}\n', mean(ts(maskbreak2)) - mean(ts(maskbreak1)));

%% Close File
fclose(fID);

% Load stock market cap/GDP data from FRED
datem = readmatrix('../RawData/DDDM01USA156NWDB.csv','Range','A:A','OutputType', 'datetime');
mktgdp = readmatrix('../RawData/DDDM01USA156NWDB.csv','Range','B:B');
% 2019 data
logdvdgdp(T-1) = log(mktgdp(end) / 100 / exp(pdm(end-1)));
logdvdgdp(T) = logdvdgdp(T-1) + divgrm(T);
for t = (T-2):-1:1
    logdvdgdp(t) = logdvdgdp(t+1) - divgrm(t);
end


% X2 mean
fID = fopen(['../tables/Table1.tex'], 'w');

fprintf(fID, '\\newcommand{\\WithoutDebtVarMeana}{%4.2f}\n', 100 * pi0);
fprintf(fID, '\\newcommand{\\WithoutDebtVarMeanb}{%4.2f}\n', 100 * y0nom_1);
fprintf(fID, '\\newcommand{\\WithoutDebtVarMeanc}{%4.2f}\n', 100 * yspr0);
fprintf(fID, '\\newcommand{\\WithoutDebtVarMeand}{%4.2f}\n', 100 * x0);
fprintf(fID, '\\newcommand{\\WithoutDebtVarMeane}{%4.2f}\n', 100 * mean(divgrm));
fprintf(fID, '\\newcommand{\\WithoutDebtVarMeanf}{%4.2f}\n', mean(logdvdgdp));
fprintf(fID, '\\newcommand{\\WithoutDebtVarMeang}{%4.2f}\n', A0m);
fprintf(fID, '\\newcommand{\\WithoutDebtVarMeanh}{%4.2f}\n', 100 * mean(deltalogtau));
fprintf(fID, '\\newcommand{\\WithoutDebtVarMeani}{%4.2f}\n', mean(log(taxrevgdp)));
fprintf(fID, '\\newcommand{\\WithoutDebtVarMeanj}{%4.2f}\n', 100 * mean(deltalogg));
fprintf(fID, '\\newcommand{\\WithoutDebtVarMeank}{%4.2f}\n', mean(log(spendgdp)));

fclose(fID);