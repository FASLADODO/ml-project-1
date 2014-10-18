function beta = leastSquares(y, tX)
% Fit a linear model using Least Squares (exact method)
  beta = (tX' * tX) \ (tX' * y);
end
