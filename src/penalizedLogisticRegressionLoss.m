function [err, gradient, hessian] = penalizedLogisticRegressionLoss(y, tX, beta, lambda)
    [err, gradient, hessian] = logisticRegressionLoss(y, tX, beta);
    
    % Never penalize beta0
    lBeta = lambda * beta;
    lBeta(1) = 0;
    
    err = err + beta' * lBeta;
    gradient = gradient + 2 * lBeta;
    hessian = hessian + 2 * lambda;
end

function [err, gradient, hessian] = logisticRegressionLoss(y, tX, beta)
    err = computeLogisticRegressionMse(y, tX, beta);
    gradient = computeLogisticRegressionGradient(y, tX, beta);
    
    sigmoid = exp(logSigmoid(tX * beta));
    S = diag(sigmoid .* (1 - sigmoid));
    
    hessian = tX' * S * tX;
end

function g = computeLogisticRegressionGradient(y, tX, beta)
% Gradient computation for the Maximum Likelihood Estimator
% of logistic regression
    n = size(y, 1);
    
	A = tX * beta;
	lSigmoid = logSigmoid(A);
	g = tX' * (exp(lSigmoid) - y) / n;
end