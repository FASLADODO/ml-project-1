function y = hybridPredictor(tX, betaClassify, betaC1, betaC2)
% Apply first a classification to distinguish two models, then a linear
% predictor with different parameters based on the class.
    classified = binaryPrediction(tX, betaClassify);
    c1 = (classified == 0);
    
    tX1 = tX(c1, :);
    tX2 = tX(~c1, :);
    
    % We applied polynomial basis expansion to learn model M1 only
    % (and never to binary variables)
    tX1 = [ones(size(tX1, 1), 1) createPoly(tX1(:, 2:36), 4) tX1(:, 37:end)];
    
    y = zeros(size(tX, 1), 1);
    y(c1) = tX1 * betaC1;
    y(~c1) = tX2 * betaC2;
end