clear; close all;

% load Treasury master files
load('crsp_treasuries_master_16_22.mat');
n=length(data.KYTREASNO);
format long

itype=data.ITYPE;
mask_itype=find(itype==option.itype);
crspid=data.KYTREASNO;

if option.itype==11 || option.itype==12
    crspid =data.KYTREASNO;
end

%coupon rate
tcouprt=data.TCOUPRT;

%number of interest payment per year
tnippy=data.TNIPPY;

%date dated by treasury (Coupon issues accrue interest beginning on the dated date)
tdatdt=data.TDATDT;
tdate_m = year(tdatdt)*100 + month(tdatdt);


%maturity date at issuance
tmatdt=data.TMATDT;
% maturity month
matdt_m=year(tmatdt)*100+month(tmatdt);

%last quote date of the month
mcaldt=data.MCALDT;
mcaldt=datenum(mcaldt);

%data date (end day of each quarter)
qtr=quarter(mcaldt);
yr=year(mcaldt);
mon=month(mcaldt);
%mcaldt_qtrs=datenum(yr,qtr*3,1);
data_date=yr*100+qtr;
data_date_m=yr*100+mon;

% time to maturity in quarters
numdaysvec=datevec(tmatdt)-datevec(mcaldt);

if option.ttm_payment==1
    ttmaturity_months = floor((numdaysvec(:,1) * 360 + numdaysvec(:,2)*30 +numdaysvec(:,3))/30);
    ttmaturity_qtrs =  numdaysvec(:,1) * 4 + floor((numdaysvec(:,2) +(-1)*(numdaysvec(:,3)<0))/3);
else
    ttmaturity_months = numdaysvec(:,1) * 12 + numdaysvec(:,2);
    ttmaturity_qtrs =  numdaysvec(:,1) * 4 + ceil((numdaysvec(:,2)/3));
end

%monthly yeild (annualized)
tmyld=data.TMYLD*365;

if option.itype==11 || option.itype==12
    tmpcyld=data.TMPCYLD;
    tmyld=2*log(tmpcyld/2+1);
end

%total amount outstanding (in millions)
tmtotout=data.TMTOTOUT;

% nominal price
price = data.TMNOMPRC;

% panel
% panel=[crspid(mask_itype) mcaldt_qtrs(mask_itype) ttmaturity_qtrs(mask_itype) tcouprt(mask_itype) tnippy(mask_itype) tmyld(mask_itype) price(mask_itype) tmtotout(mask_itype) issuance date(month)];
panel=[crspid(mask_itype) mcaldt(mask_itype) data_date(mask_itype) data_date_m(mask_itype) ttmaturity_qtrs(mask_itype) ttmaturity_months(mask_itype) tcouprt(mask_itype) tnippy(mask_itype) tmyld(mask_itype) price(mask_itype) tmtotout(mask_itype) tdate_m(mask_itype) matdt_m(mask_itype)];

% position in the panel;
pos.crspid = 1;
pos.mcaldt = 2; %quote date
pos.date_qr= 3; % quote date (quarters)
pos.date_m = 4; % quote date (month)
pos.ttm_qr = 5;
pos.ttm_m  = 6;
pos.couprt = 7;
pos.tnippy = 8;
pos.tmyld  = 9;
pos.price  = 10;
pos.tmout  = 11;
pos.tdate_m  = 12; % issuance date
pos.matdt_m =13;

tcusip = var(2:end,4);
tcusip = tcusip(mask_itype);

mask_nan=find(isnan(panel(:,pos.tmout))~=1); %total amount oustanding missing
panel=panel(mask_nan,:);
tcusip=tcusip(mask_nan);
mask_nan=find(isnan(panel(:,pos.price))~=1); % price missing
panel=panel(mask_nan,:);
tcusip=tcusip(mask_nan);

mask_nan=find(isnan(panel(:,pos.tmout))==0); % total amount outstanding is zero
panel=panel(mask_nan,:);
tcusip=tcusip(mask_nan);

% replace tnippy NaN as 2 for coupon note
mask_nan=find(panel(:,pos.tnippy)==0);
if option.itype==4 || option.itype==8
    for j=1:length(mask_nan)
        panel(mask_nan(j),pos.tnippy)=0;
    end
else
    for j=1:length(mask_nan)
        panel(mask_nan(j),pos.tnippy)=2;
    end
end

mask_nan=find(isnan(panel(:,pos.couprt))==1);
for j=1:length(mask_nan)
    panel(mask_nan(j),pos.couprt)=0;
end

clearvars mask_nan

%%%%%%%%%%%%%%%%%%
%%%% new issuance by changing the market value of each crspid
%%%% crspid identifies the bonds with same maturity (could be issued
%%%% through different auctions)
id=unique(panel(:,1));
new_issue_cusip=zeros(length(panel),1);

for jj=1:length(id)
    mask_id=find(panel(:,1)==id(jj));
    panel_ni_temp=panel(mask_id, :);
    tmout_ni_temp=panel_ni_temp(2:end,pos.tmout)-panel_ni_temp(1:end-1,pos.tmout);
    tmout_ni_temp=[panel_ni_temp(1,pos.tmout);tmout_ni_temp];
    new_issue_cusip(mask_id,:)=tmout_ni_temp;
end
clear jj mask_id

%%%%%%%%%%%%%%%%%
%%%% double use TCUSIP to identify
id2=unique(tcusip);
if ~isequal(id2, {'0XX'}) ~=0
    for jj=1:length(id2)
        if ~isequal(id2(jj), {'0XX'})
            mask_id=find(strcmp(tcusip, id2(jj)));
            panel_ni_temp2=panel(mask_id, :);
            tmout_ni_temp2=panel_ni_temp2(2:end,pos.tmout)-panel_ni_temp2(1:end-1,pos.tmout);
            tmout_ni_temp2=[panel_ni_temp2(1,pos.tmout);tmout_ni_temp2];
            new_issue_tcusip(mask_id,:)=tmout_ni_temp2;
        end
    end
end

p_payment_cusip=zeros(length(id),2); % principal payment plus the last coupon payment

for jj=1:length(id)
    mask_id=find(panel(:,1)==id(jj));
    panel_matdt_temp=panel(mask_id, pos.matdt_m);
    temp=panel(mask_id, pos.tmout);
    %     p_payment_cusip(jj,:)=[unique(panel(mask_id, pos.matdt_m)) temp(end)];
    coupon_matdt_temp=temp.*(panel(mask_id, pos.couprt)./panel(mask_id, pos.tnippy))/100;
    coupon_matdt_temp=coupon_matdt_temp(end);
    if isnan(coupon_matdt_temp)==1
        coupon_matdt_temp=0;
    end
    p_payment_cusip(jj,:)=[unique(panel(mask_id, pos.matdt_m)) temp(end)+coupon_matdt_temp ];
end

panel=[panel(:,1:12) new_issue_cusip panel(:,13)];
pos.newissue_m  = 13;

clear jj

pos.matdt_m =14;

%%%%%%%%%%%%%%%%%%%%%%%
% only subset of the data
panel = sortrows(panel,pos.date_m);
sampleyear = round(panel(:,pos.date_m)/100);
masktemp1 = min(find(sampleyear >=beginyear));
masktemp2 = max(find(sampleyear <=endyear));
panel = panel(masktemp1:masktemp2,:);
clearvars masktemp1 masktemp2 sampleyear