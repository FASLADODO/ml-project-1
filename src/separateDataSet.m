function [X_m1, y_m1, X_m2, y_m2] = separateDataSet(X, y, featureNo)
% TODO : learn the threshold automatically (or even learn a full
% classifier)
    threshold = 1.2;
    ms = X(:,featureNo) > threshold;
    
    X_m1 = X(ms,:);
    y_m1 = y(ms);
    
    X_m2 = X(~ms,:);
    y_m2 = y(~ms);
end