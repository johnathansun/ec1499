clear; close all;

% Form pd ratio for the vw-market portfolio
crspvwdata  = xlsread('crsp_vwret_2021.xlsx','','a2:f1141'); %1926.1-2020.12 (1140 months)
crspvw = crspvwdata(:,1);
crspvw_nodivs= crspvwdata(:,2);
crspdate = crspvwdata(:,3)*10000+crspvwdata(:,4)*100+crspvwdata(:,5);
exindexm(1,:) = 1;
for tt= 2:11
    exindexm(tt,:) = exindexm(tt-1,:).*(1+crspvw_nodivs(tt,:)); %1140 by 1
end
adjpriceindm  = exindexm;

% Get the start date of the time series
start_year = 1926;
start_month = 1;

%% Smooth out strange dividend events
% One-time adjustment
% Microsoft; Nov 2004
adj_flag1 = 1;
t1 = 12*(2004-start_year) + (11-start_month) + 1;
% Time Warner Cable; Mar 2009
adj_flag2 = 1;
t2 = 12*(2009-start_year) + (3-start_month) + 1;
% 2012 Q4; Oct 2012
adj_flag3 = 1;
t3 = 12*(2012-start_year) + (10-start_month) + 1;

% Generate divqtrm and exindexm from crspvw and crspvw_nodivs
for t = 12:length(crspvw)
    divqtrm(t,:)      = (crspvw(t,:)-crspvw_nodivs(t,:)).*exindexm(t-1,:);
    exindexm(t,:)     = exindexm(t-1,:).*(1+crspvw_nodivs(t,:));
end

% One-time adjustment
% Microsoft case
if adj_flag1 == 1
    % Nov 2004
    t = t1;
    % Calculate the "special dividend"
    spcdiv_2004 = divqtrm(t,:) - mean(divqtrm(t-12:t-1,:));
    spcdivadj_2004 = spcdiv_2004/12;
    %%% Adjust dividends
    divqtrm(t,:) = divqtrm(t,:)-spcdiv_2004;
    divqtrm(t-1:t+10,:) = divqtrm(t-1:t+10,:)+spcdivadj_2004;
    %%% Adjust crspvw
    % Calculate new series of p_t + d_t
    adjcumindexm = exindexm + divqtrm;
    % Calculate adjusted crspvw
    adjcrspvw = [1;adjcumindexm(2:end)./exindexm(1:end-1)-1];
    crspvw(t-1:t+10,:)=adjcrspvw(t-1:t+10,:);
end
% Time Warner Cable case
if adj_flag2 == 1
    % Mar 2009
    t = t2;
    % Calculate the "special dividend"
    spcdiv_2009 = divqtrm(t,:) - mean(divqtrm(t-12:t-1,:));
    spcdivadj_2009 = spcdiv_2009/12;
    %%% Adjust dividends
    divqtrm(t,:) = divqtrm(t,:)-spcdiv_2009;
    divqtrm(t-2:t+9,:) = divqtrm(t-2:t+9,:)+spcdivadj_2009;
    %%% Adjust crspvw
    % Calculate new series of p_t + d_t
    adjcumindexm = exindexm + divqtrm;
    % Calculate adjusted crspvw
    adjcrspvw = [1;adjcumindexm(2:end)./exindexm(1:end-1)-1];
    crspvw(t-2:t+9,:)=adjcrspvw(t-2:t+9,:);
end
% 2012 case
if adj_flag3 == 1
    % Oct 2012
    t = t3;
    % Calculate the "special dividend" for each month
    spcdiv_2012_1 = divqtrm(t,:) - mean(divqtrm(t-12:t-1,:));
    spcdiv_2012_2 = divqtrm(t+1,:) - mean(divqtrm(t-12+1:t-1+1,:));
    spcdiv_2012_3 = divqtrm(t+2,:) - mean(divqtrm(t-12+2:t-1+2,:));
    spcdivadj_2012_1 = spcdiv_2012_1/12;
    spcdivadj_2012_2 = spcdiv_2012_2/12;
    spcdivadj_2012_3 = spcdiv_2012_3/12;
    %%% Adjust dividends
    divqtrm(t,:) = divqtrm(t,:)-spcdiv_2012_1;
    divqtrm(t+1,:) = divqtrm(t+1,:)-spcdiv_2012_2;
    divqtrm(t+2,:) = divqtrm(t+2,:)-spcdiv_2012_3;
    divqtrm(t:t+11,:) = divqtrm(t:t+11,:)+spcdivadj_2012_1;
    divqtrm(t+1:t+11+1,:) = divqtrm(t+1:t+11+1,:)+spcdivadj_2012_2;
    divqtrm(t+2:t+11+2,:) = divqtrm(t+2:t+11+2,:)+spcdivadj_2012_3;
    %%% Adjust crspvw
    % Calculate new series of p_t + d_t
    adjcumindexm = exindexm + divqtrm;
    % Calculate adjusted crspvw
    adjcrspvw = [1;adjcumindexm(2:end)./exindexm(1:end-1)-1];
    crspvw(t:t+11+2,:)=adjcrspvw(t:t+11+2,:);
end

%% Further adjustments
for t = 12: length(crspvw)
    adjcapgainm(t,:)  = crspvw_nodivs(t,:) - (divqtrm(t-1,:)/12+divqtrm(t-2,:)/12+divqtrm(t-3,:)/12+divqtrm(t-4,:)/12+divqtrm(t-5,:)/12+divqtrm(t-6,:)/12+divqtrm(t-7,:)/12+divqtrm(t-8,:)/12+divqtrm(t-9,:)/12+divqtrm(t-10,:)/12+divqtrm(t-11,:)/12 - 11*divqtrm(t,:)/12)./exindexm(t-1,:);
    adjpriceindm(t,:) = adjpriceindm(t-1,:).*(1+adjcapgainm(t,:));
    adjdivyieldm(t,:) = crspvw(t,:)- adjcapgainm(t,:);
    divpricem(t,:)   = adjdivyieldm(t,:).*adjpriceindm(t-1,:)./adjpriceindm(t,:);
    divgrm(t,:)      = ((1+crspvw(t,:))./divpricem(t-1,:))./(1+(divpricem(t,:).^(-1)))-1;
end

begin_y = 19260130;
begin_mask = find(crspdate==begin_y);
dpm     = log(divpricem(begin_mask:end,:));     % log pd ratio %1926.1-2019.12
pdm     = -dpm;               % log pd ratio
pdm_q   = pdm(3:3:end)-log(3); % sample
pdm_y   = pdm(12:12:end)-log(12); % annualized log pd ratio % 1926-2019 (december)
logdivgrm = log(1+divgrm(begin_mask:end,:)); % log monthly dividend growth %1929.1-2019.12
logdivgr_q = logdivgrm(1:3:end-2)+logdivgrm(2:3:end-1)+logdivgrm(3:3:end);
logdivgr_y = logdivgrm(1:12:end-11)+logdivgrm(2:12:end-10)+logdivgrm(3:12:end-9)+logdivgrm(4:12:end-8)+logdivgrm(5:12:end-7)+logdivgrm(6:12:end-6)...
    + logdivgrm(7:12:end-5)+logdivgrm(8:12:end-4)+logdivgrm(9:12:end-3)+logdivgrm(10:12:end-2)+logdivgrm(11:12:end-1)+logdivgrm(12:12:end);

%% Final results
newresdata_adj = [pdm_q logdivgr_q];

ttime=[1926:2020]';
xlswrite('MaindatafileA_Feb2021_longsample.xlsx',pdm_y(4:end), 'VAR','N2:N93');
xlswrite('MaindatafileA_Feb2021_longsample.xlsx',logdivgr_y(4:end), 'VAR','O2:O93');

