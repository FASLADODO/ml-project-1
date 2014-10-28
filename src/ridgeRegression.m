function beta = ridgeRegression(y, tX, alpha)
% Learn model parameters beta using Ridge Regression
% alpha: gradient descent step size

  % TODO: learn lambda using k-fold cross-validation
  lambda = 1;

  gramMatrix = (tX' * tX);
  l = lambda * eye(size(gramMatrix));
  beta = (gramMatrix + l) \ (tX' * y);

end
