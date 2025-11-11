clear; close all;

% load Treasury master files
filename='treasuries_crsp_46_20.xlsx';

T=readtable(filename);
TMTOTOUT = T.TMTOTOUT;
TMPUBOUT = T.TMPUBOUT;
TMNOMPRC = T.TMNOMPRC;

MCALDT = T.MCALDT;
year = year(MCALDT);
month = month(T.MCALDT);
date=year*100+month;
mkt_out=zeros((2020-1946+1)*4,1);
mkt_ph_out=zeros((2020-1946+1)*4,1);

% aggregation
for y = 1946:2020
    for q=1:4
        tempd=y*100+q*3;
        mask_date=find(date==tempd);
        i=(y-1946)*4+q;
        date_crsp(i)=y*100+q;
        mkt_out(i)=nansum(TMTOTOUT(mask_date).*TMNOMPRC(mask_date)/100);
        mkt_ph_out(i)=nansum(TMPUBOUT(mask_date).*TMNOMPRC(mask_date)/100);
    end
end

%% annual data
filename='nominal_aggr_debt_1946_2020_annual.xls';
var1={'yyyy'};
xlswrite(filename,var1,'','A1');
xlswrite(filename,unique(year),'','A2:A76');
var2={'mkt_tot_out'};
var3={'mkt_public_out'};
xlswrite(filename,var2,'','B1');
xlswrite(filename,mkt_out(4:4:end),'','B2:B76');


% read annual gdp data
gdp_46_20=xlsread('NIPATable 1.1.5.xls','Sheet0','T8:CP8');
gdp=[gdp_46_20(1,:) ]'*1000;
holdings_gdp=mkt_out(4:4:end)./gdp;
gdp_46=gdp(1);

% debt gdp ratio for 1946 using average GDP
holdings_gdp_1946 = holdings_gdp(1);
clearvars holdings_gdp
gdp_47_10=xlsread('NIPATable1.1.5_2020.xls','Sheet0','C8:IV8');
gdp_10_20=xlsread('NIPATable1.1.5_2020.xls','Sheet1','C8:AR8');
gdp=[gdp_47_10(1,1:end) gdp_10_20(1,:)]'*1000;
holdings_gdp=mkt_out(5:end)./gdp;
holdings_gdp = [holdings_gdp_1946; holdings_gdp(4:4:end)];
gdp_annual = [gdp_46; gdp(4:4:end)];

var4={'gdp (millions)'};
var5={'holdings_to_gdp'};
xlswrite(filename,var4,'','C1');
xlswrite(filename,gdp_annual,'','C2:C76');
xlswrite(filename,var5,'','D1');
xlswrite(filename,holdings_gdp,'','D2:D76');


