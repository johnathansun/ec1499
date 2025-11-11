function [Psi, Sig] = run_VAR(X2, N, T, inflpos, y1pos, ysprpos, gdppos, divgrpos, pdpos, dtpos, dgpos, ...
        coint_tax, coint_spending, coint_div)

    z_var = X2(2:end, :);
    z_varlag = X2(1:end - 1, :);
    Y = z_var;
    X = z_varlag;

    Psi = zeros(N, N);

    for i = [inflpos y1pos ysprpos gdppos divgrpos pdpos dtpos dgpos]
        regr = ols(Y(:, i), [ones(T - 1, 1), X(:, :)]);
        c(i) = regr.beta(1);
        Psi(i, :) = regr.beta(2:end)';
        eps(:, i) = Y(:, i) - c(i) - X * Psi(i, :)';
    end

    Psi(coint_tax, :) = Psi(dtpos, :);
    Psi(coint_spending, :) = Psi(dgpos, :);
    Psi(coint_div, :) = Psi(divgrpos, :);
    Psi(coint_tax, coint_tax) = Psi(coint_tax, coint_tax) + 1;
    Psi(coint_spending, coint_spending) = Psi(coint_spending, coint_spending) + 1;
    Psi(coint_div, coint_div) = Psi(coint_div, coint_div) + 1;

    Sigma = cov(eps(:, [inflpos y1pos ysprpos gdppos divgrpos pdpos dtpos dgpos]));
    tmp = chol(Sigma, 'lower');
    Sig = zeros(N, N);
    Sig([inflpos y1pos ysprpos gdppos divgrpos], [inflpos y1pos ysprpos gdppos divgrpos]) = ...
        tmp(1:5, 1:5);
    Sig(pdpos, [inflpos y1pos ysprpos gdppos divgrpos pdpos]) = tmp(6, 1:6);
    Sig(dtpos, [inflpos y1pos ysprpos gdppos divgrpos pdpos dtpos]) = tmp(7, 1:7);
    Sig(dgpos, [inflpos y1pos ysprpos gdppos divgrpos pdpos dtpos dgpos]) = tmp(8, 1:8);

    Sig(coint_div, :) = Sig(divgrpos, :);
    Sig(coint_tax, :) = Sig(dtpos, :);
    Sig(coint_spending, :) = Sig(dgpos, :);
