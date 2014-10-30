function [X_m1, y_m1, X_m2, y_m2] = separateDataSet(X, y, featureNo)
% TO DO : learning threshold automatically
    threshold = 1.2;
    ms = X(:,featureNo) > threshold;
    
    % linear model
    X_m1 = X(ms,:);
    y_m1 = y(ms);
    % other model
    X_m2 = X(~ms,:);
    y_m2 = y(~ms);