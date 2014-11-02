function [Xnew, ynew] = removeOutliers(X, y, nDeviations)
% nDeviations: number of standard deviations allowed away from the median
% in the input data
    mediansX = ones(length(X), 1) * median(X);
    deviationsX = ones(length(X), 1) * std(X);
    % Find indices of the outliers
    outlier = abs(X - mediansX) > nDeviations * deviationsX;
    t = sum(outlier,2) > 0;
    Xnew = X(~t,:);
    ynew = y(~t);
end