clear; close all;
load crsp_treasuries_master_46_20.mat;

crspid=data(:,3); % CRSP unique ID
tdatdt=var(2:end,5); % date dated by Treasury
tmatdt=var(2:end,6); % maturity date at time of issue 
tcouprt=data(:,8); % coupon rate
mcaldt=var(2:end,23); % last quote date in the month
bid=data(:,24);
ask=data(:,25);
tmnomprc=data(:,26); % nominal price
tmretnua=data(:,30); % monthly unadjusted return
tmyld=data(:,31); % monthly series of promised daily yield
tmduratn=data(:,32); % monthly series of Macaulay's duration
tmtotout=data(:,33); % total outstanding (face value in millions)
tmpubout=data(:,34); % publicly held outstanding (face value in millions)
tmpcyld=data(:,35); % semi-annual yield
tmretnxs=data(:,36); % monthly excess return
itype=data(:,16); % issuance type

% convert the quote date to month
date_master=split(mcaldt,"/");
date_master=str2double(date_master);
datadate=date_master(:,3)*100+date_master(:,1);
datem=unique(date_master(:,3)*100+date_master(:,1)); %194701-202012;

% calculate market value of each bond, and set the market value to zero if
% return is missing
out_case = 1;

switch out_case
    case 1
        mktval = tmnomprc.*tmtotout;
    case 2
        mktval = tmnomprc.*tmpubout;
end

mktval(find(isnan(mktval)==1))=0;

% calculate aggregate portfolio weight and weighted average return
for i=1:length(datem)
    mask.temp=find(datadate==datem(i));
    mktval_temp=mktval(mask.temp);
    duration_temp=tmduratn(mask.temp);
    mask_missingr=find( duration_temp~=-99 & isnan(duration_temp)~=1 );
    if mean(mktval_temp)~=0
    mktval_total(i)=sum(mktval_temp(mask_missingr));
    
    fieldname=strcat('w',num2str(datem(i)));
    weight.aggr.(fieldname)=mktval_temp(mask_missingr)./mktval_total(i);
    duration.aggr(i)=sum(weight.aggr.(fieldname).*duration_temp(mask_missingr));
    clear mask.temp mktval_temp yld_temp returnx_temp returnua_temp duration_temp mask_missingr fieldname i mktval_total_temp
    else
        duration.aggr(i)=nanmean(duration_temp(mask_missingr));
    end
end

save duration_aggr_46_20.mat duration datem
xlswrite('duration_aggregate_46_20.xls',[datem duration.aggr']);