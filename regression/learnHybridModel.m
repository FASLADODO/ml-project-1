function allBetas = learnHybridModel(y, tX, threshold)
% Learn a classifier and two separate models from the regression dataset
% threshold: Value of the output used to separate the two models
% OUTPUT
%     allBetas cell array (Dx3)
%       One set of parameters for each model (classification, first model, second model)

    allBetas = {};

    % Model separation
    % We make the assumption that two distinct models can be used to explain
    % the output: one with "constant", high value of y; and another model.
    % We learn a classifier to separate the two.

    [betaClassifier, tX1, y1, tX2, y2] = separateDataSet(y, tX, threshold);
    allBetas{1} = betaClassifier;
    
    % We now easily spot some outliers in the second model
    outliers2 = y2 > 7200;
    tX2 = tX2(~outliers2, :);
    y2 = y2(~outliers2);

    % Learn model M1
    % The first model is not obvious and should be learnt
    % using a ML technique.

    % Basis function expansion
    % It allows us to fit a more complex model, but might produce overfitting.
    % This is why we use ridge regression.
    % We found that polynomials of degree 4 produced the best tradeoff.
    % Warning: do not apply basis extension to binary variables!
    % Note that this transformation must be applied to all new input data at
    % the time of prediction.
    tX1 = [ones(size(tX1, 1), 1) createPoly(tX1(:, 2:36), 4) tX1(:, 37:end)];
    
    % Ridge regression uses its own k-fold cross validation to select lambda
    seed = randi(10000);
    k = 5;
    lambdas = logspace(0, 2, 100);
    allBetas{2} = ridgeRegressionAuto(y1, tX1, k, lambdas, seed);

    % Learn model M2
    % As observed, the second model is simply a constant
    allBetas{3} = [mean(y2); zeros(size(tX2, 2) - 1, 1)];
end