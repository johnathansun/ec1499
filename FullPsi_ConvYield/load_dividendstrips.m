% The data file itself contains more details on data construction
data_dvd = xlsread('../RawData/APdata.xlsx', 'VAR', 'a2:ay294');

Tstart_dvd = 109; % start sample in 1974.Q1 (1=1947.Q1)
Tend_dvd = 292; % end sample in 2019.Q4
T_dvd = Tend_dvd - Tstart_dvd + 1;

date_dvd = data_dvd(Tstart_dvd:Tend_dvd, 1);

PDm_strip_4q_data = data_dvd(Tstart_dvd:Tend_dvd, 21); % PD ratio on next 4 quarters of dividend strips
PDm_strip_8q_data = data_dvd(Tstart_dvd:Tend_dvd, 23); % PD ratio on next 8 quarters of dividend strips
sharestrip_4q_data = data_dvd(Tstart_dvd:Tend_dvd, 25); % PD ratio on next 4 quarters of dividend strips
sharestrip_8q_data = data_dvd(Tstart_dvd:Tend_dvd, 27); % PD ratio on next 8 quarters of dividend strips
% D is the sum of the past 12 months divided by 4

dvd.PDm_strip_4q_data = [ones(27, 1) * NaN; PDm_strip_4q_data((1:(T_dvd / 4)) * 4) / 4; NaN];
dvd.PDm_strip_8q_data = [ones(27, 1) * NaN; PDm_strip_8q_data((1:(T_dvd / 4)) * 4) / 4; NaN];
dvd.sharestrip_4q_data = [ones(27, 1) * NaN; sharestrip_4q_data((1:(T_dvd / 4)) * 4); NaN];
dvd.sharestrip_8q_data = [ones(27, 1) * NaN; sharestrip_8q_data((1:(T_dvd / 4)) * 4); NaN];
% now data are 1947 to 2020

% Risk premia on dividend futures
dvd.portfdivfuturereturn_data = (0.0041 + 0.0059 + 0.0067 + 0.0072 + 0.0084 + 0.0090 + 0.0095) * 1200/7;
