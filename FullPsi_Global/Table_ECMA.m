clear; close all;
load ('MAT/result_step3.mat');

yielddata_GSW = readtable('../RawData/feds200628.csv');
yielddate = yielddata_GSW.Date;
yieldyear = year(yielddate);
yieldmonth = month(yielddate);
yieldday = day(yielddate);

year = unique(yieldyear);

for i = 1:length(year)
    mask = find(yieldyear == year(i));
    mask = mask(end);
    yield01(i) = nanmean(yielddata_GSW.SVENY01(mask));
    yield04(i) = nanmean(yielddata_GSW.SVENY04(mask));
    yield05(i) = nanmean(yielddata_GSW.SVENY05(mask));
end

yield01 = yield01' / 100;
yield04 = yield04' / 100;
yield05 = yield05' / 100;

bondexcessreturn_model = (-1) * (Api(5) + Bpi(:, 5)' * X2(1:end - 1, :)') + (Api(4) + Bpi(:, 4)' * X2(2:end, :)') - ((-1) * Api(1) - Bpi(:, 1)' * X2(1:end - 1, :)');
bondexcessreturn_data = -4 * yield04(2:end) + 5 * yield05(1:end - 1) - yield01(1:end - 1);

erp_rmse_global = sqrt(mean((equityriskpremTV_model' - equityriskpremTV_data') .^ 2)); % in percentage
brp_rmse_global = sqrt(mean((100 * bondriskpremTV_model' - 100 * bondriskpremTV_data') .^ 2)); % in percentage
bexret_rmse_global = sqrt(nanmean(nanmean((100 * bondexcessreturn_model' - 100 * bondexcessreturn_data') .^ 2)));

Nom_error = 100 * (kron(ones(length(X2t), 1), -Api(tau)' ./ tau) - ((Bpi(:, tau)' ./ kron(tau', ones(1, N))) * X2t')' - yielddata);
nomyield_rmse_global = sqrt(nanmean(nanmean([Nom_error] .^ 2)));

Real_error = 100 * (kron(ones(length(X2t), 1), -A(tipstau)' ./ tipstau) - ((B(:, tipstau)' ./ kron(tipstau', ones(1, N))) * X2t')' - tipsdata);
realyield_rmse_global = sqrt(nanmean(nanmean([Real_error] .^ 2)));

% pd ratio
pd_rmse_global = sqrt(nanmean((exp(A0m + I_pdm' * X2t')' - PDm_model) .^ 2));

fID = fopen(['../tables/Table_globalECMA.tex'], 'w');
fprintf(fID, '\\newcommand{\\erprmseg}{%4.2f}\n', erp_rmse_global);
fprintf(fID, '\\newcommand{\\brprmseg}{%4.2f}\n', brp_rmse_global);
fprintf(fID, '\\newcommand{\\bexretrmseg}{%4.2f}\n', bexret_rmse_global);
fprintf(fID, '\\newcommand{\\nomyieldrmseg}{%4.2f}\n', nomyield_rmse_global);
fprintf(fID, '\\newcommand{\\realyieldrmseg}{%4.2f}\n', realyield_rmse_global);
fprintf(fID, '\\newcommand{\\pdmrmseg}{%4.2f}\n', pd_rmse_global);

%% model without global risk factors
load('../FullPsi/MAT/result_step3.mat');

bondexcessreturn_model = (-1) * (Api(5) + Bpi(:, 5)' * X2(1:end - 1, :)') + (Api(4) + Bpi(:, 4)' * X2(2:end, :)') - ((-1) * Api(1) - Bpi(:, 1)' * X2(1:end - 1, :)');

erp_rmse = sqrt(mean((equityriskpremTV_model' - equityriskpremTV_data') .^ 2)); % in percentage
brp_rmse = sqrt(mean((100 * bondriskpremTV_model' - 100 * bondriskpremTV_data') .^ 2)); % in percentage
bexret_rmse = sqrt(nanmean(nanmean((100 * bondexcessreturn_model' - 100 * bondexcessreturn_data') .^ 2)));

Nom_error = 100 * (kron(ones(length(X2t), 1), -Api(tau)' ./ tau) - ((Bpi(:, tau)' ./ kron(tau', ones(1, N))) * X2t')' - yielddata);
nomyield_rmse = sqrt(nanmean(nanmean([Nom_error] .^ 2)));

Real_error = 100 * (kron(ones(length(X2t), 1), -A(tipstau)' ./ tipstau) - ((B(:, tipstau)' ./ kron(tipstau', ones(1, N))) * X2t')' - tipsdata);
realyield_rmse = sqrt(nanmean(nanmean([Real_error] .^ 2)));

% pd ratio
pd_rmse = sqrt(nanmean((exp(A0m + I_pdm' * X2t')' - PDm_model) .^ 2));

fprintf(fID, '\\newcommand{\\erprmse}{%4.2f}\n', erp_rmse);
fprintf(fID, '\\newcommand{\\brprmse}{%4.2f}\n', brp_rmse);
fprintf(fID, '\\newcommand{\\bexretrmse}{%4.2f}\n', bexret_rmse);

fprintf(fID, '\\newcommand{\\nomyieldrmse}{%4.2f}\n', nomyield_rmse);
fprintf(fID, '\\newcommand{\\realyieldrmse}{%4.2f}\n', realyield_rmse);
fprintf(fID, '\\newcommand{\\pdmrmse}{%4.2f}\n', pd_rmse);

fclose(fID);
