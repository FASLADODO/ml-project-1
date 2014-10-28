function beta = ridgeRegression(y, tX, lambda)
% Learn model parameters beta using Ridge Regression
% lambda: penalization parameter
    gramMatrix = (tX' * tX);
    l = lambda * eye(size(gramMatrix));
    beta = (gramMatrix + l) \ (tX' * y);
end
