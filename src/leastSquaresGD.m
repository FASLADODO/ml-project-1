function beta = leastSquaresGD(y, tX, alpha)
% Fit a linear model using Least Squares (using gradient descent)
% alpha: gradient descent step size

  % algorithm parametesfor maximum iterations and convergence
  maxIters = 1000;
  convergence_th = 0; % convergence threshold

  % initialize
  beta = zeros(size(tX, 2), 1);

  L_last = 0;

  % iterate
  for k = 1:maxIters
    % compute gradient 
    g = computeGradient(y, tX, beta);

    % compute cost
    L = computeRmse(y, tX, beta);

    % update beta
    beta = beta - alpha * g;

    % check convergence
    if abs(L_last - L) < convergence_th
    	break;
    end 

    L_last = L;

  end



end
