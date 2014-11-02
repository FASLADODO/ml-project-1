function beta = leastSquaresGD(y, tX, alpha)
% Fit a linear model using Least Squares (using gradient descent)
% alpha: gradient descent step size
    maxIters = 10000;
    % Convergence threshold
    epsilon = 1e-6;

    % Initialization
    beta = zeros(size(tX, 2), 1);
    previousL = 0;

    % iterate
    for k = 1:maxIters
        g = computeGradient(y, tX, beta);
        L = computeRmse(y, tX * beta);
        % Update beta (descent rule)
        beta = beta - alpha * g;

        % Check for convergence
        if abs(previousL - L) < epsilon
            break;
        end 

        previousL = L;
    end;
    
    %fprintf('Gradient descent stopped after %d iterations.\n', k);
end
