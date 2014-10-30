function yHat = binaryPrediction(tX, beta)
% Generate the predictions for the given data examples
% and the learnt model beta
    n = size(tX, 1);
    A = tX * beta;
    probabilities = exp(logSigmoid(A));
    yHat = zeros(n, 1);
    yHat(probabilities > 0.5) = 1;
end