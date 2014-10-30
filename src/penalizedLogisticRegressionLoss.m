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
    err = computeMse(y, tX, beta);
    gradient = computeGradient(y, tX, beta);
    
    sigmoid = exp(logSigmoid(tX * beta));
    S = diag(sigmoid .* (1 - sigmoid));
    
    hessian = tX' * S * tX;
end

function err = computeMse(y, tX, beta)
    n = size(y, 1);
    
	e = y - binaryPrediction(tX, beta);
	err = (e' * e) / (2 * n);
end