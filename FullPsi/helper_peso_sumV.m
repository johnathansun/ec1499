function PDpeso_model = helper_peso_sumV(phi, X2t, y0nom_1, g0, x0, pi0, nu, Sigma, Sig, Psi, L0, L1, ...
        I_y1, I_gdp, I_gstrip, striphorizon)

    Apeso(1) = log(1 - phi) -y0nom_1 + g0 + x0 + pi0 + ...
        1/2 * (I_gstrip)' * Sigma * (I_gstrip) - (I_gstrip)' * Sig * L0 + ...
        (-L0' + (I_gstrip)' * Sig) * (nu * I_gdp);

    Bpeso(:, 1) = ((I_gstrip)' * Psi - I_y1' ...
        - (I_gstrip' * Sig + nu * I_gdp') * L1)';

    PDpeso_model = exp(Apeso(1) + Bpeso(:, 1)' * X2t')';

    for j = 1:(striphorizon - 1)
        Apeso(j + 1) = log(1 - phi) + Apeso(j) -y0nom_1 + g0 + x0 + pi0 + ...
            1/2 * (I_gstrip + Bpeso(:, j))' * Sigma * (I_gstrip + Bpeso(:, j)) - (I_gstrip + Bpeso(:, j))' * Sig * L0 + ...
            (-L0' + (I_gstrip + Bpeso(:, j))' * Sig) * (nu * I_gdp);

        Bpeso(:, j + 1) = ((I_gstrip + Bpeso(:, j))' * Psi - I_y1' ...
            - ((I_gstrip + Bpeso(:, j))' * Sig + nu * I_gdp') * L1)';

        PDpeso_model = PDpeso_model + exp(Apeso(j + 1) + Bpeso(:, j + 1)' * X2t')';
    end

end
