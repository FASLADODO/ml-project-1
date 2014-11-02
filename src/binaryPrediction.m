function [yHat, pHat] = binaryPrediction(tX, beta)
% Generate the predictions for the given data examples
% and the learnt model beta
    n = size(tX, 1);
    A = tX * beta;
    pHat = exp(logSigmoid(A));
    yHat = zeros(n, 1);
    yHat(pHat > 0.5) = 1;
end