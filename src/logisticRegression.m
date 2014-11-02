function beta = logisticRegression(y, tX, alpha)
% Fit a linear model using Logistic Regression (using Newton's method)
% alpha: step size
    if(nargin < 3)
        alpha = 0.1;
    end;

    beta = penLogisticRegression(y, tX, alpha, 0);
end


