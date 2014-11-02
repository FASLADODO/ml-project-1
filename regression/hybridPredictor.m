function y = hybridPredictor(tX, betaClassify, betaC1, betaC2)
% Apply first a classification to distinguish two models, then a linear
% predictor with different parameters based on the class.
    classified = binaryPrediction(tX, betaClassify);
    c1 = (classified == 0);
    
    y = zeros(size(tX, 1), 1);
    y(c1) = tX(c1, :) * betaC1;
    y(~c1) = tX(~c1, :) * betaC2;
end