clear; close all;

% Load Data
LoadData_Benchmark;
LoadData_WithDebt;
% Run ConvYield/ImportConvYield.R in R to clean convenience yield data

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% VAR Estimation and Descriptive Stats
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Table 1
% TablePsi.tex
Table_Basics;
% Figure 2
Plot_TG_Beta;
% Figure 3
Plot_FiscalImpulseResponses;
% Figure 4
Plot_CashFlowForecasts;

% Figure B.1a ../Code/FullPsi/figures/pic-cy.pdf
% Figure B.1b ../Code/FullPsi/figures/pic-sgnrev.pdf
Plot_ConvYield;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Upper Bound Exercise
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Figure 5 ../code/FullPsi/figures/cs30.pdf
% Figure 6a ../code/FullPsi/figures/cs-pd30.pdf
clear; close all;
output_rp = 0.03;
Campbell_Shiller;

% Model Variants

% Figure C.5a ../code/FullPsi/figures/cs-WithDebt30.pdf
% Figure C.5b ../code/FullPsi/figures/cs-WithDebt25.pdf
clear; close all;
output_rp = 0.03;
Campbell_Shiller_WithDebt;
clear; close all;
output_rp = 0.025;
Campbell_Shiller_WithDebt;

% Figure C.6a ../code/FullPsi/figures/cs_longsample_30.pdf
% Figure C.6b ../code/FullPsi/figures/cs_longsample_25.pdf
clear; close all;
output_rp = 0.03;
Campbell_Shiller_LongSample;
clear; close all;
output_rp = 0.025;
Campbell_Shiller_LongSample;

% Figure C.7 ../code/FullPsi/figures/cs_cy_27.pdf
Campbell_Shiller_ConvYield;

% Figure 7  ../code/FullPsi/cs_summary.pdf
Plot_Campbell_Shiller;

% Figure 6b ../code/FullPsi/figures/cs_EvaluateAtDetrended.pdf
Campbell_Shiller_EvaluateAtDetrended;

% Figure C.1 ../code/FullPsi/figures/pic-wedge_cyclicality_CS.pdf
Campbell_Shiller_WedgeCyclicality;

% Figure C.2 ../code/FullPsi/figures/cs25.pdf
clear; close all;
output_rp = 0.025;
Campbell_Shiller;

% Figure C.3a ../code/FullPsi/figures/cs_vary_rp.pdf
% Figure C.3b ../code/FullPsi/figures/cs_rp015_pdx.pdf
Campbell_Shiller_Vary_Risk_Premium;

% Figure C.4a ../code/FullPsi/figures/cs-pd-TVRP.pdf
% Figure C.4b ../code/FullPsi/figures/cs-pv-delta-TVRP.pdf
Campbell_Shiller_TVRiskPremia;

% TableCampbell_Shiller.tex
Table_Campbell_Shiller;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Main Asset Pricing Model
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Estimation
% Step 1, Figure E.1 - E.6
main_step1;

% Step 2
main_step2;

% Step 3, Figure E.7 - E.10
main_step3;

% Figure 1, 8 - 11, Figure D.1, Figure E.11
Plot_Benchmark;
Plot_Peso;

% TableRates.tex, TableLambda.tex
Table_Benchmark;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Cointegration Tests
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Cointegration_Johansen;
% also run Phillips-Ouliaris.R in R

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Robustness: Asset Pricing Model with Global VAR
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Figure E.12
run('../FullPsi_Global/Campbell_Shiller_Global.m');
% Figure E.13
run('../FullPsi_Global/main_step1.m');
run('../FullPsi_Global/main_step2.m');
run('../FullPsi_Global/main_step3.m');
% Print Table
run('../FullPsi_Global/Table_ECMA.m');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Robustness: Asset Pricing Model with Convenience Yield
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
run('../FullPsi_ConvYield/main_step1.m');
run('../FullPsi_ConvYield/main_step2.m');
run('../FullPsi_ConvYield/main_step3.m');
