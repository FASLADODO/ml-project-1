function correlatedVariables = computeFeaturesCorrelations(X)

    correlations = corr(X);

    figure;
    imagesc(correlations);
    colorbar;

    [corrI, corrJ] = find(abs(correlations) > 0.4);
    idx = (corrI - corrJ > 0);
    correlatedVariables = [corrI(idx) corrJ(idx)];
    for i = 1:length(correlatedVariables)
        correlatedVariables(i, 3) = correlations(correlatedVariables(i, 1), correlatedVariables(i, 2));
    end;
    correlatedVariables = sortrows(correlatedVariables, [-3, 1, 2]);
    % We obtain the highest correlation coefficients and the corresponding
    % input variables indices
end