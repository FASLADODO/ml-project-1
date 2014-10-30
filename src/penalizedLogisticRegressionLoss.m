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

function err = computeLogisticRegressionMse(y, tX, beta)
    n = size(y, 1);
    
	%e = y - binaryPrediction(tX, beta);
	%err = (e' * e) / (2 * n);
    
    lSigmoid = logSigmoid(tX * beta);
    logLikelihood = sum(y .* lSigmoid + (1 - y) .* log(1 - exp(lSigmoid)));
    err = - logLikelihood / n;
    
end

function g = computeLogisticRegressionGradient(y, tX, beta)
% Gradient computation for the Maximum Likelihood Estimator
% of logistic regression
    n = size(y, 1);
    
	A = tX * beta;
	lSigmoid = logSigmoid(A);
	g = tX' * (exp(lSigmoid) - y) ./ n;
end