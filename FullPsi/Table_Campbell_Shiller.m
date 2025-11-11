clear; close all;

%% Saving parameter estimates to automatically update the paper
fID = fopen(['../tables/TableCampbell_Shiller.tex'], 'w');

load('MAT/CS30.mat');

fprintf(fID, '\\newcommand{\\CSrminusg}{%4.2f}\n', (y0nom_1 - x0 - pi0) * 100);
fprintf(fID, '\\newcommand{\\CSexpretminusg}{%4.2f}\n', (y0nom_1 + yspr0 + output_rp - x0 - pi0) * 100);
fprintf(fID, '\\newcommand{\\CSrealr}{%4.2f}\n', (y0nom_1 - pi0) * 100);

fprintf(fID, '\\newcommand{\\CSxbar}{%4.2f}\n', (x0) * 100);
fprintf(fID, '\\newcommand{\\CSexpret}{%4.2f}\n', (y0nom_1 + yspr0 + output_rp - pi0) * 100);
fprintf(fID, '\\newcommand{\\CSkappax}{%4.2f}\n', k1x);
fprintf(fID, '\\newcommand{\\CSpdx}{%4.2f}\n', exp(pxbar));
fprintf(fID, '\\newcommand{\\CSpdxround}{%4.0f}\n', exp(pxbar));
fprintf(fID, '\\newcommand{\\CStauminusg}{%4.2f}\n', mean(taxrevgdp - spendgdp) * 100);
fprintf(fID, '\\newcommand{\\CSupper}{%4.2f}\n', upper * 100);
fprintf(fID, '\\newcommand{\\CSupperchange}{%4.2f}\n', (1 / exp(pxbar) - mean(taxrevgdp - spendgdp)) * 100);
fprintf(fID, '\\newcommand{\\CSupperto}{%4.2f}\n', 1 / exp(pxbar) * 100);

fprintf(fID, '\\newcommand{\\CSmeantau}{%4.2f}\n', exp(mean(log(taxrevgdp))) * 100);
fprintf(fID, '\\newcommand{\\CStautoday}{%4.2f}\n', taxrevgdp(T) * 100);
fprintf(fID, '\\newcommand{\\CSmeang}{%4.2f}\n', exp(mean(log(spendgdp))) * 100);
fprintf(fID, '\\newcommand{\\CSgtoday}{%4.2f}\n', spendgdp(T) * 100);
fprintf(fID, '\\newcommand{\\CSgtrendval}{%4.2f}\n', (exp(log(spendgdp(T)) - X2(T, coint_spending))) * 100);
fprintf(fID, '\\newcommand{\\CSwedge}{%4.2f}\n', (mean(gdebt) - mean(s)) * 100);

load('MAT/CS_ConvYield.mat');
fprintf(fID, '\\newcommand{\\CSmeanconvenience}{%4.2f}\n', mean(cy) * 100);
fprintf(fID, '\\newcommand{\\CSmeanconvenienceHalf}{%4.2f}\n', mean(cy / 2) * 100);
fprintf(fID, '\\newcommand{\\CSK}{%4.2f}\n', mean(cyk) * 100);
fprintf(fID, '\\newcommand{\\CSKgdp}{%4.2f}\n', exp(pxbar) * mean(cyk) * 100);
fprintf(fID, '\\newcommand{\\CStaukminusgConvYield}{%4.2f}\n', mean(taxrevgdp - spendgdp) * 100);
fprintf(fID, '\\newcommand{\\CSrealrConvYield}{%4.2f}\n', (y0nom_1 - pi0) * 100);
fprintf(fID, '\\newcommand{\\CSpdxConvYield}{%4.2f}\n', exp(pxbar));
fprintf(fID, '\\newcommand{\\CSupperConvYield}{%4.2f}\n', upper * 100);

load('MAT/CS_LongSample30.mat');
fprintf(fID, '\\newcommand{\\CSpdxLongSample}{%4.2f}\n', exp(pxbar));
fprintf(fID, '\\newcommand{\\CStauminusgLongSample}{%4.2f}\n', mean(taxrevgdp - spendgdp) * 100);
fprintf(fID, '\\newcommand{\\CSupperLongSample}{%4.2f}\n', upper * 100);
fprintf(fID, '\\newcommand{\\CSupperchangeLongSample}{%4.2f}\n', (1 / exp(pxbar) - mean(taxrevgdp - spendgdp)) * 100);
fprintf(fID, '\\newcommand{\\CSuppertoLongSample}{%4.2f}\n', 1 / exp(pxbar) * 100);

load('../FullPsi_Global/MAT/CS_Global.mat')
fprintf(fID, '\\newcommand{\\CSglobalwedge}{%4.2f}\n', (mean(gdebt) - mean(s)) * 100);

fclose(fID);
