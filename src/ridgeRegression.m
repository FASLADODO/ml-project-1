function beta = ridgeRegression(y, tX, lambda)
% Learn model parameters beta using Ridge Regression
% lambda: penalization parameter
    gramMatrix = (tX' * tX);
    eigenValues = eye(size(gramMatrix));
    % preventing from lifting beta0 value
    eigenValues(:,1) = zeros(size(eigenValues,1),1);
    l = lambda * eigenValues;
    beta = (gramMatrix + l) \ (tX' * y);
end
