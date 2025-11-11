function [obj, FF] = solve_step3(param, N, T, Psi, Sig, I_pi, I_gdp, I_y1, I_yspr, I_pdm, I_divgrm, I_dg, divgrpos, ...
        mu_m, k1m, r0_m, A0m, striphorizon, y0nom_1, y0nom_5, pi0, x0, g0, X2t, tau, yielddata, tipstau, tipsdata, dvd, lev, eps2)

    L0 = zeros(N, 1);
    L1 = zeros(N, N);

    i = 1;
    L0(I_pi == 1) = param(i); i = i + 1;
    L0(I_yspr == 1) = param(i); i = i + 1;
    L0(I_gdp == 1) = param(i); i = i + 1;
    L0(I_divgrm == 1) = param(i); i = i + 1;
    L0(I_pdm == 1) = param(i); i = i + 1;

    L1(I_pi == 1, I_pi == 1) = param(i); i = i + 1;
    L1(I_pi == 1, I_y1 == 1) = param(i); i = i + 1;
    L1(I_pi == 1, I_yspr == 1) = param(i); i = i + 1;
    L1(I_pi == 1, I_gdp == 1) = param(i); i = i + 1;
    L1(I_yspr == 1, 1:N) = param(i:i + N - 1); i = i + N;
    L1(I_gdp == 1, 1:N) = param(i:i + N - 1); i = i + N;
    L1(I_pdm == 1, 1:N) = param(i:i + N - 1); i = i + N;

    FF = [];

    % Discipline the eigenvalues of the risk neutral transition matrix Psi - Sig * L1
    if (max(abs(eig(Psi - Sig * L1))) >= 0.995)
        FF = [FF 1e10];
    end

    maxeigim = max(abs(imag(eig(Psi - Sig * L1))));
    eigim_baseline = .5;
    FF = [FF max(maxeigim - eigim_baseline, 0) * 1e7];

    %% arbitrage-free definition of real risk-free rate
    y0_1 = y0nom_1 - pi0 - .5 * I_pi' * (Sig * Sig') * I_pi + I_pi' * Sig * L0;

    %% Bond pricing
    A = zeros(striphorizon, 1);
    B = zeros(N, striphorizon);
    Api = zeros(striphorizon, 1);
    Bpi = zeros(N, striphorizon);
    Am = zeros(striphorizon, 1);
    Bm = zeros(N, striphorizon);

    A(1) =- y0_1;
    B(:, 1) =- (I_y1' - I_pi' * Psi + I_pi' * Sig * L1)';

    Api(1) = -y0nom_1;
    Bpi(:, 1) = -I_y1';

    for j = 1:striphorizon + 50
        Api(j + 1) =- y0nom_1 + Api(j) + .5 * Bpi(:, j)' * (Sig * Sig') * Bpi(:, j) - Bpi(:, j)' * Sig * L0;
        Bpi(:, j + 1) = (Bpi(:, j)' * Psi - I_y1' - Bpi(:, j)' * Sig * L1)';

        A(j + 1) =- y0_1 + A(j) + .5 * B(:, j)' * (Sig * Sig') * B(:, j) - B(:, j)' * Sig * (L0 - Sig' * I_pi);
        B(:, j + 1) = ((I_pi + B(:, j))' * Psi - I_y1' - (I_pi + B(:, j))' * Sig * L1)';
    end

    %% Equity strip
    I_equitystrip = I_divgrm + I_gdp + I_pi;
    Am(1) = mu_m + x0 - y0nom_1 +pi0 + .5 * (I_equitystrip)' * Sig * Sig' * (I_equitystrip) - (I_equitystrip)' * Sig * L0;
    Bm(:, 1) = ((I_equitystrip)' * Psi - I_y1' - (I_equitystrip)' * Sig * L1)';
    PDm_model = exp(Am(1) + Bm(:, 1)' * X2t')';

    for j = 1:striphorizon - 1
        Am(j + 1) = Am(j) + mu_m + x0 - y0nom_1 +pi0 + .5 * (I_equitystrip + Bm(:, j))' * Sig * Sig' * (I_equitystrip + Bm(:, j)) - (I_equitystrip + Bm(:, j))' * Sig * L0;
        Bm(:, j + 1) = ((I_equitystrip + Bm(:, j))' * Psi - I_y1' - (I_equitystrip + Bm(:, j))' * Sig * L1)';
        PDm_model = PDm_model + exp(Am(j + 1) + Bm(:, j + 1)' * X2t')';
    end

    %% GDP strip
    I_xstrip = (I_gdp + I_pi);
    Ax(1) = -y0nom_1 + x0 + pi0 + .5 * (I_xstrip)' * (Sig * Sig') * (I_xstrip) - (I_xstrip)' * Sig * L0;
    Bx(:, 1) = ((I_xstrip)' * Psi - I_y1' - (I_xstrip)' * Sig * L1)';
    PDx_model0 = exp(Ax(1) + Bx(:, 1)' * X2t')';

    for j = 1:(striphorizon - 1)
        Ax(j + 1) = Ax(j) - y0nom_1 + x0 + pi0 + 0.5 * (I_xstrip + Bx(:, j))' * (Sig * Sig') * (I_xstrip + Bx(:, j)) - (I_xstrip + Bx(:, j))' * Sig * L0;
        Bx(:, j + 1) = (-I_y1' + (I_xstrip + Bx(:, j))' * Psi - (I_xstrip + Bx(:, j))' * Sig * L1)';
        PDx_model0 = PDx_model0 + exp(Ax(j + 1) + Bx(:, j + 1)' * X2t')';
    end

    %% For final refinement only: value of G strips don't blow up
    I_gstrip = (I_dg + I_gdp + I_pi);
    Ag(1) = -y0nom_1 + g0 + x0 + pi0 + .5 * (I_gstrip)' * (Sig * Sig') * (I_gstrip) - (I_gstrip)' * Sig * L0;
    Bg(:, 1) = ((I_gstrip)' * Psi - I_y1' - (I_gstrip)' * Sig * L1)';
    PDg_model0 = exp(Ag(1))';

    for j = 1:(striphorizon - 1)
        Ag(j + 1) = Ag(j) -y0nom_1 + g0 + x0 + pi0 + .5 * (I_gstrip + Bg(:, j))' * (Sig * Sig') * (I_gstrip + Bg(:, j)) - (I_gstrip + Bg(:, j))' * Sig * L0;
        Bg(:, j + 1) = ((I_gstrip + Bg(:, j))' * Psi - I_y1' - (I_gstrip + Bg(:, j))' * Sig * L1)';
        PDg_model0 = PDg_model0 + exp(Ag(j + 1))';
    end

    if (PDg_model0 > 1e3)
        FF = [FF 1e10];
    end

    %% No arbitrage restrictions on stock return
    % (I_equitystrip+k1m*I_pdm)'*Sig*L0 == (r0_m + pi0 - y0nom_1 + 0.5*(aa)*(aa)')
    % (I_equitystrip+k1m*I_pdm)'*Sig*L1(:,i) == ((I_equitystrip+k1m*I_pdm)'*Psi -I_pdm' -I_y1')(i)
    penalty = 1e4;
    ind = [1:(divgrpos - 1), (divgrpos + 1):N];

    aa = (I_equitystrip + k1m * I_pdm)' * Sig;
    cc = ((I_equitystrip + k1m * I_pdm)' * Psi -I_pdm' -I_y1');
    FF_new = penalty * (L0(divgrpos) - (((r0_m + pi0 - y0nom_1 + 0.5 * (aa) * (aa)') - aa(ind) * L0(ind)) / aa(divgrpos))) ^ 2;
    FF = [FF FF_new];

    penalty = 5e3;
    for i = 1:N
        FF_new = penalty * (L1(divgrpos, i) - ((cc(i) - aa(ind) * L1(ind, i)) / aa(divgrpos))) ^ 2;
        FF = [FF FF_new];
    end

    %% No arbitrage restrictions on bond yields
    % -Api(5)/5 == y0nom_5
    % -Bpi(:,5)'/5 == target
    penalty = 1e6;
    FF_newa = penalty * ((-Api(5) / 5 - y0nom_5) * 100) ^ 2;
    FF = [FF FF_newa];

    penalty = 5e6;
    target = zeros(1, N);
    target(2) = 1;
    target(3) = 1;
    FFtmp = penalty * (-Bpi(:, 5)' / 5 - target) .^ 2;
    FF = [FF, FFtmp];

    %% Match Stock Price/Dividend Ratio
    penalty = 2e4;
    FF_new = penalty * nanmean((exp(A0m + I_pdm' * X2t')' - PDm_model) .^ 2);
    FF = [FF FF_new];

    %% Match Unlevered GDP Risk Premium
    penalty = 1e5;
    FF_new = 0;
    lookup_horizon = [1:10, 20:10:90];

    for i = 1:length(lookup_horizon)
        tt = lookup_horizon(i);
        gdp_rp = (-Ax(tt) ./ tt - y0nom_1 + (x0 + pi0)) * 100;
        stock_rp = (-Am(tt)' ./ tt - y0nom_1 + (mu_m + x0 + pi0)) * 100;

        FF_new = FF_new + penalty * (100 - tt) / 100 * (gdp_rp - stock_rp * lev) .^ 2;
    end

    FF = [FF FF_new];

    %% Match Dividend Strip
    PDm_strip_4q = exp(Am(1) + Bm(:, 1)' * X2t')';
    PDm_strip_8q = exp(Am(1) + Bm(:, 1)' * X2t')' + exp(Am(2) + Bm(:, 2)' * X2t')';
    sharestrip_4q = PDm_strip_4q ./ PDm_model;
    sharestrip_8q = PDm_strip_8q ./ PDm_model;

    penalty = 1e3;
    FF_new = penalty * nanmean([dvd.PDm_strip_4q_data - PDm_strip_4q] * 100) .^ 2;
    FF = [FF FF_new];
    FF_new = penalty * nanmean([dvd.PDm_strip_8q_data - PDm_strip_8q] * 100) .^ 2;
    FF = [FF FF_new];
    penalty = 1e4;
    FF_new = penalty * nanmean([dvd.sharestrip_4q_data - sharestrip_4q] * 100) .^ 2;
    FF = [FF FF_new];
    FF_new = penalty * nanmean([dvd.sharestrip_8q_data - sharestrip_8q] * 100) .^ 2;
    FF = [FF FF_new];

    %% Risk premia on dividend futures
    hor = 7;
    divfuturereturn = zeros(hor, T);
    divfuturereturn(1, 2:end) = exp(0 ...
        - (Am(1) - Api(1)) - (Bm(:, 1) - Bpi(:, 1))' * X2t(1:end - 1, :)' ...
        +x0 + mu_m + pi0 + (I_gdp + I_divgrm + I_pi)' * X2t(2:end, :)') -1;

    divfuturereturn(2:hor, 2:end) = exp(Am(1:hor - 1) - Api(1:hor - 1) + (Bm(:, 1:hor - 1) - Bpi(:, 1:hor - 1))' * X2t(2:end, :)' ...
        - (Am(2:hor) - Api(2:hor)) - (Bm(:, 2:hor) - Bpi(:, 2:hor))' * X2t(1:end - 1, :)' ...
        +x0 + mu_m + pi0 + (I_gdp + I_divgrm + I_pi)' * X2t(2:end, :)') -1;
    portfdivfuturereturn_model = 100 * mean(mean(divfuturereturn(1:hor, 57:68), 2));
    % avg of first 7 annual strip returns, avg from 2003-2014
    penalty = 5e4;
    FF_new = penalty * (dvd.portfdivfuturereturn_data - portfdivfuturereturn_model) ^ 2;
    FF = [FF FF_new];

    %% Match Yield Curve
    % Pricing nominal bond yields of maturities stored in tau
    penalty = 1e6;
    Nom_error = 100 * (kron(ones(length(X2t), 1), -Api(tau)' ./ tau) - ((Bpi(:, tau)' ./ kron(tau', ones(1, N))) * X2t')' - yielddata);
    FF_new = penalty * nanmean([Nom_error] .^ 2);
    FF = [FF FF_new];

    % Pricing real bond yields
    penalty = 1e5;
    Real_error = 100 * (kron(ones(length(X2t), 1), -A(tipstau)' ./ tipstau) - ((B(:, tipstau)' ./ kron(tipstau', ones(1, N))) * X2t')' - tipsdata);
    FF_new = penalty * nanmean([Real_error] .^ 2);
    FF = [FF FF_new];

    %% Good deal bounds
    nomshortrate = y0nom_1 + I_y1' * X2t'; % 1 by T
    L = kron(L0, ones(1, T)) + L1 * X2t'; % N by T

    eps_pos = [1:5 7 8 10];
    eps_orig = Sig(eps_pos, eps_pos) \ eps2(:, eps_pos)'; % eps2 are VAR resids, they have covariance Sigma, eps_orig has covariance matrix Identity
    eps_orig = eps_orig';
    eps_orig = [zeros(1, N - 3); eps_orig]; % N by T; shocks in period 1 set to zero)\eps2(:,1:(N-3))';

    mnom(1) = -nomshortrate(1) - .5 * L(:, 1)' * L(:, 1) - L(eps_pos, 1)' * eps_orig(1, :)';
    mreal(1) = mnom(1) + pi0 + I_pi' * X2t(1, :)';

    for t = 2:T
        mnom(t) = -nomshortrate(t - 1) - .5 * L(:, t - 1)' * L(:, t - 1) -L(eps_pos, t - 1)' * eps_orig(t, :)';
        mreal(t) = mnom(t) + pi0 + I_pi' * X2t(t, :)';
    end

    FF_new = 2e5 * (exp(max(std(mnom) - 2, 0)) - 1);
    FF = [FF FF_new];

    %% Regularity conditions
    penalty = 1e5;
    lookup_horizon = [100, 200, 400, 700, 1e3, 2e3, 3e3, 4e3];

    % Forcing nominal yield + GDP risk premium > nominal GDP growth
    for i = 1:length(lookup_horizon)
        tt = lookup_horizon(i);
        gdp_expret = (-Ax(tt) / tt - y0nom_1 + (x0 + pi0)) - Api(tt) / tt;
        FF_new = penalty * abs(min((gdp_expret - (pi0 + x0)) * 100, 0));
        FF = [FF FF_new];
    end

    % Forcing the nominal-real spread to stay above average inflation
    for i = 1:length(lookup_horizon)
        tt = lookup_horizon(i);
        FF_new = penalty * abs(min(-100 * Api(tt) / tt + 100 * A(tt) / tt -pi0 * 100, 0));
        FF = [FF FF_new];
    end

    % Forcing the bond risk premium to stay below the GDP risk premium
    for i = 1:length(lookup_horizon)
        tt = lookup_horizon(i);
        gdp_rptt = (-Ax(tt) ./ tt - y0nom_1 + (x0 + pi0)) * 100;
        nom_rptt = (-Api(tt) ./ tt - y0nom_1) * 100;
        FF_new = penalty * abs(min(gdp_rptt - nom_rptt, 0));
        FF = [FF FF_new];
    end

    % Forcing bond return vol to be bounded by 30%
    for i = 1:length(lookup_horizon)
        tt = lookup_horizon(i);
        nombondret = 100 * ((Api(tt - 1) + Bpi(:, tt - 1)' * X2t(2:end, :)') - ...
            (Api(tt) + Bpi(:, tt)' * X2t(1:end - 1, :)'));
        std_nombondret = std(nombondret);
        FF_new = penalty * abs(min(30 - std_nombondret, 0));
        FF = [FF FF_new];
    end

    % Forcing bond risk premium to be bounded by 2%
    for i = 1:length(lookup_horizon)
        tt = lookup_horizon(i);
        nombondrp = 100 * (- Api(tt) / tt + Api(1));
        FF_new = penalty * abs(min(2 - nombondrp, 0));
        FF = [FF FF_new];
    end

    obj = nansum(abs(FF));
