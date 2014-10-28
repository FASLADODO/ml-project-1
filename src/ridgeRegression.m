function beta = ridgeRegression(y, tX, alpha)
% Learn model parameters beta using Ridge Regression
% alpha: gradient descent step size

  gramMatrix = (tX' * tX);
  l = alpha * eye(size(gramMatrix));
  beta = (gramMatrix + l) \ (tX' * y);

end
