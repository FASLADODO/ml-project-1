function [Xnew ynew] = removeOutliers(X, y, threshold)
    mediansX = ones(length(X), 1) * median(X);
    deviationsX = ones(length(X), 1) * std(X);
    outlier = abs(X - mediansX) > threshold*deviationsX; % Find outlier idx
    t = sum(outlier,2) > 0;
    Xnew = X(~t,:);
    ynew = y(~t);
end