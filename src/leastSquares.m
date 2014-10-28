function beta = leastSquares(y, tX)
% Fit a linear model using Least Squares (normal equations)
  beta = (tX' * tX) \ (tX' * y);
end
