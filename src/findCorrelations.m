function [correlatedVariables, correlations] = findCorrelations(selector, X, y)
% selector: function(x) => boolean selecting the features with respect to
% their correlation, e.g threshold = 0.5; selector = @(x) abs(x) > threshold;
    
    % if y is passed as parameter we are looking for correlations between
    % input and output variables otherwise we are looking for correlations
    % between features
    if (nargin < 3)
        correlations = corr(X);
    else
        correlations = corr(X,y);
    end
     

    [corrI, corrJ] = find(selector(correlations));
    idx = (corrI - corrJ > 0);
    correlatedVariables = [corrI(idx) corrJ(idx)];
    for i = 1:length(correlatedVariables)
        correlatedVariables(i, 3) = correlations(correlatedVariables(i, 1), correlatedVariables(i, 2));
    end;
    correlatedVariables = sortrows(correlatedVariables, [-3, 1, 2]);
end


