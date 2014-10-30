function beta = logisticRegression(y, tX, alpha)
% Fit a linear model using Logistic Regression (using Newton's method)
% alpha: step size
    beta = penLogisticRegression(y, tX, alpha, 0);
end


