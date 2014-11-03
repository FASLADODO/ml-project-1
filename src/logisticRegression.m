function beta = logisticRegression(y, tX, alpha)
% Fit a linear model using Logistic Regression (using Newton's method)
% alpha: step size to use for Newton's method
    if(nargin < 3)
        alpha = 1e-3;
    end;

    beta = penLogisticRegression(y, tX, alpha, 0);
end


