clear; close all;

beginyear=1929;
endyear=2022;

dateq=zeros(beginyear-endyear+1*4,1);
datem=zeros(beginyear-endyear+1*12,1);

for i=beginyear:endyear
    for q=1:4
        dateq((i-beginyear)*4+q)=i*100+q;
    end
    for m=1:12
        datem((i-beginyear)*12+m)=i*100+m;
    end
end
panel_coup_newissue_m_total=zeros(length(datem),4);
panel_mkt_aggr_total=zeros(length(datem),2);
mkv_aggr_total=zeros(length(dateq),1);

panel_newissue_tcusip_m_total=zeros(length(datem),1);

% use time to maturity to identify coupons/principals that mature this
% month/quarter
option.ttm_payment=1;

p_payment_m_total=zeros(length(datem),2);

for i=1:12
    option.itype=i;
    position_const_loaddata_2022;

    % identify zero coupon bond
    mask_zeros=find(panel(:,pos.couprt)==0);
    panel_zeros=panel(mask_zeros,:);

    % identify coupon bonds (strips)
    mask_coupon=find(panel(:,pos.couprt)~=0);
    panel_coupon=panel(mask_coupon,:);
    strip_master=[];
    % identify unique quote date (day)
    date_id=unique(panel_coupon(:, pos.mcaldt));
    dateqr_id=unique(panel_coupon(:, pos.date_qr));
    datem_id=unique(panel_coupon(:, pos.date_m));

    % p_payment_m=zeros(length(datem_id),1);
    coup_payment_m=zeros(length(datem_id),1);
    tmout_newissue_m=zeros(length(datem_id),1);
    mkv_newissue_m=zeros(length(datem_id),1);
    mkt_aggr_m=zeros(length(datem_id),1);

    if (length(panel_coupon)>0)
        for j=1:length(datem_id)
            mask_date=find(panel_coupon(:,pos.date_m)==datem_id(j));
            panel_coupon_temp=panel_coupon(mask_date,:);

            % identify coupon payment month
            % number of months before maturity
            m_coupon=(round(panel_coupon_temp(:,pos.date_m)/100)-round(panel_coupon_temp(:,pos.tdate_m)/100))*12 + ...
                (panel_coupon_temp(:,pos.date_m)-round(panel_coupon_temp(:,pos.date_m)/100)*100) - ...
                (panel_coupon_temp(:,pos.tdate_m)-round(panel_coupon_temp(:,pos.tdate_m)/100)*100);
            m_coupon(isnan(m_coupon))=1/3;
            idx_coupon=~mod(m_coupon./(12./panel_coupon_temp(:,pos.tnippy)),1);
            mask_idx_coupon=find(idx_coupon==1);
            coup_payment_m(j)=sum(panel_coupon_temp(mask_idx_coupon,pos.tmout).*panel_coupon_temp(mask_idx_coupon, pos.couprt)./panel_coupon_temp(mask_idx_coupon, pos.tnippy))/100;

            tmout_newissue_m(j)=sum(panel_coupon_temp(:, pos.newissue_m));
            mkv_newissue_m(j)=sum(panel_coupon_temp(:, pos.newissue_m).*panel_coupon_temp(:,pos.price))./100;
            mkt_aggr_m(j)=sum(panel_coupon_temp(:, pos.tmout).*panel_coupon_temp(:, pos.price))/100;
        end
    end

    clear j k
    clear mask_date panel_coupon_temp mask_coup m_coupon idx_coupon mask_idx_coupon mkv_newissue_temp tmout_newissue_temp %tmout_newissue_tcusip_m

    %% zero coupon bond
    if length(panel_zeros)>0
        datem_id0=unique(panel_zeros(:, pos.date_m));
        % p_payment_zeros_m=zeros(length(datem_id0),1);
        tmout_newissue_zeros_m=zeros(length(datem_id0),1);
        mkv_newissue_zeros_m=zeros(length(datem_id0),1);
        mkt_aggr_zeros_m=zeros(length(datem_id0),1);

        for j=1:length(datem_id0)
            mask_date=find(panel_zeros(:,pos.date_m)==datem_id0(j));
            panel_zeros_temp=panel_zeros(mask_date,:);

            tmout_newissue_zeros_m(j)=sum(panel_zeros_temp(:, pos.newissue_m));
            mkv_newissue_zeros_m(j)=sum(panel_zeros_temp(:, pos.newissue_m).*panel_zeros_temp(:,pos.price))./100;

            mkt_aggr_zeros_m(j)=sum(panel_zeros_temp(:, pos.tmout).*panel_zeros_temp(:, pos.price))/100;
        end

    else
        tmout_newissue_zeros_m=zeros(length(datem_id),1);
        tmout_newissue_tcusip_zeros_m= zeros(length(datem_id),1);
        mkv_newissue_zeros_m=zeros(length(datem_id),1);
        tmout_newissue_zeros_m_v2=zeros(length(datem_id),1);
        mkv_newissue_zeros_m_v2=zeros(length(datem_id),1);
        mkt_aggr_zeros_m=zeros(length(datem_id),1);
        tmout_newissue_zeros_temp_v2=zeros(length(datem_id),1);
    end

    % aggregate
    if (length(panel_zeros)>0 && length(panel_coupon)==0)
        tmout_newissue_m=tmout_newissue_zeros_m;
        mkv_newissue_m=mkv_newissue_zeros_m;
        mkt_aggr_m=mkt_aggr_zeros_m;
        coup_payment_m=zeros(length(datem_id0),1);
        
        panel_coup_newissue=[datem_id0 coup_payment_m tmout_newissue_m mkv_newissue_m];
        panel_mkt_aggr=[datem_id mkt_aggr_m];

        datem_range=find(ismember(datem,panel_coup_newissue(:,1)));
        panel_coup_newissue_m_total(datem_range,2:end)=panel_coup_newissue_m_total(datem_range,2:end)+panel_coup_newissue(:,2:end);
        panel_mkt_aggr_total(datem_range,2)=panel_mkt_aggr_total(datem_range,2)+mkt_aggr_m;

        panel_coup_newissue_m_total(:,1)=datem;
    else
        tmout_newissue_m=tmout_newissue_m+tmout_newissue_zeros_m;
        mkv_newissue_m=mkv_newissue_m+mkv_newissue_zeros_m;
        mkt_aggr_m=mkt_aggr_m+mkt_aggr_zeros_m;

        panel_coup_newissue=[datem_id coup_payment_m tmout_newissue_m mkv_newissue_m];
        panel_mkt_aggr=[datem_id mkt_aggr_m];

        datem_range=find(ismember(datem,panel_coup_newissue(:,1)));
        panel_coup_newissue_m_total(datem_range,2:end)=panel_coup_newissue_m_total(datem_range,2:end)+panel_coup_newissue(:,2:end);
        panel_mkt_aggr_total(datem_range,2)=panel_mkt_aggr_total(datem_range,2)+mkt_aggr_m;
    end

    % principal payment
    p_payment_m=zeros(length(datem),1);

    mask_matdt=find(ismember(p_payment_cusip(:,1), datem)); %find the maturity date within the sample period from 1947-2019. This is called valid principal payment date
    p_payment_date=unique(p_payment_cusip(mask_matdt,1)); % identify the valid pincipal payment date

    % for each valid principal payment date, we aggregate the total amout
    % outstand (face value) + the last coupon payment of the matured issues.
    % p_payment_cusip includes the coupon payment
    for idp=1:length(datem)
        p_payment_m(idp)=sum(p_payment_cusip(find(p_payment_cusip(:,1)==datem(idp)),2));
    end

    p_payment_m_total(:,2:end)=p_payment_m_total(:,2:end)+p_payment_m(:,1);

    filename1 = sprintf('%s_%d','MAT/newissue_month',i);
    save (filename1, 'panel_coup_newissue', 'panel_mkt_aggr', 'p_payment_m')
    clearvars i
end

panel_m_total=[datem' p_payment_m_total(:,2) panel_coup_newissue_m_total(:,2:end)];

save 'MAT/panel_coup_newissue_tcusip_total_month.mat' panel_newissue_tcusip_m_total panel_m_total datem dateq beginyear endyear


%% convert monthly total to quarterly and annually total
load 'MAT/panel_coup_newissue_tcusip_total_month.mat'

for i=beginyear:endyear
    for m=1:12
        if floor(m/3)==m/3
            dq((i-beginyear)*12+m,1)=i*100+(m/3);
            datey((i-beginyear)*12+m,1)=i;
        else
            dq((i-beginyear)*12+m,1)=i*100+floor(m/3)+1;
            datey((i-beginyear)*12+m,1)=i;
        end
    end
end

for i=beginyear:endyear
    for q=1:4
        newissue_total_qr((i-beginyear)*4+q,1)=sum(panel_m_total(find(dq==i*100+q),5));
        newissue_total_face_qr((i-beginyear)*4+q,1)=sum(panel_m_total(find(dq==i*100+q),4));
        coupon_total_qr((i-beginyear)*4+q,1)=sum(panel_m_total(find(dq==i*100+q),3));
        principal_total_qr((i-beginyear)*4+q,1)=sum(panel_m_total(find(dq==i*100+q),2));
    end
    newissue_total_yr((i-beginyear)+1,1)=sum(panel_m_total(find(datey==i),5));
    newissue_total_face_yr((i-beginyear)+1,1)=sum(panel_m_total(find(datey==i),4));
    coupon_total_yr((i-beginyear)+1,1)=sum(panel_m_total(find(datey==i),3));
    principal_total_yr((i-beginyear)+1,1)=sum(panel_m_total(find(datey==i),2));
end

net_po=coupon_total_qr+principal_total_qr-newissue_total_qr;

%%
aggrdebt=xlsread('nominal_aggr_debt_1929_2022_annual.xls');
load('MAT/mkt_out_crsp_1929_2022.mat');

% annual mkt value
mkt_out = mkt_out(min(find(round(date_crsp/100)==beginyear)):end);
mkt_out_yr=mkt_out(4:4:end);

if beginyear >= 1929
    panel_coup_newissue_total_yr=[(beginyear:1:endyear)' principal_total_yr(1:end) coupon_total_yr(1:end) newissue_total_face_yr(1:end) newissue_total_yr(1:end) mkt_out_yr];
else
    panel_coup_newissue_total_yr=[(1929:1:endyear)' principal_total_yr(1:end) coupon_total_yr(1:end) newissue_total_face_yr(1:end) newissue_total_yr(1:end) mkt_out_yr];
end

panel_coup_newissue_total_qr=[dateq' principal_total_qr coupon_total_qr newissue_total_face_qr newissue_total_qr mkt_out];
save 'MAT/panel_coup_newissue_total.mat' panel_coup_newissue_total_qr panel_coup_newissue_total_yr


%%
filename='panel_coup_newissue_total_yr_1929_2022.xls';
var1={'yyyy'};
xlswrite(filename,var1,'','A1');
xlswrite(filename,panel_coup_newissue_total_yr,'','A2:G95');
var2={'Principal Payment'};
var3={'Coupon Payment'};
var4={'New Issuance (Millions) Face'};
var5={'New Issuance (Millions) MKV'};
var6={'Aggr Mkt Value (Millions)'};
xlswrite(filename,var2,'','B1');
xlswrite(filename,var3,'','C1');
xlswrite(filename,var4,'','D1');
xlswrite(filename,var5,'','E1');
xlswrite(filename,var6,'','F1');