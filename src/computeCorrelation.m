function correlationWithOutput = computeCorrelation(X,y)

    correlations = corr(X,y);

    figure;
    imagesc(correlations);
    colorbar;

    [corrI, corrJ] = find(abs(correlations) > 0.4);
    idx = (corrI - corrJ > 0);
    correlationWithOutput = [corrI(idx) corrJ(idx)];
    
    for i = 1:length(correlationWithOutput)
        correlationWithOutput(i, 3) = correlations(correlationWithOutput(i, 1), correlationWithOutput(i, 2));
    end;
    
    correlationWithOutput = sortrows(correlationWithOutput, [-3, 1, 2]);
    
end