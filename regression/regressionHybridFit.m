% Regression fitting
addpath(genpath('./data'), genpath('../data'));
addpath(genpath('./src'), genpath('../src'));

clear;
load('regression.mat');
% Our goal is to output values for this input
XtoPredict = X_test;

% We change the seed (and thus the train / test split)
% at each run to check the stability of our results
seed = randi(10000);
[X, y, X_test, y_test] = split(y_train, X_train, 0.8, seed);

%% Preprocessing

% Normalize features (except the discrete ones)
notNormalized = X(:,1:35);
[X(:,1:35), X_test(:,1:35)] = normalized(notNormalized, X_test(:,1:35));
[~, XtoPredict(:,1:35)] = normalized(notNormalized, XtoPredict(:,1:35));
clear notNormalized;

% We perform dummy variables encoding on categorical variables only
categoricalVariables = [36 38 40 43 44];
X = dummyEncoding(X, categoricalVariables);
X_test = dummyEncoding(X_test, categoricalVariables);
XtoPredict = dummyEncoding(XtoPredict, categoricalVariables);

N = length(y);
tX = [ones(N, 1) X];
tX_test = [ones(length(y_test), 1) X_test];
tXtoPredict = [ones(size(XtoPredict, 1), 1) XtoPredict];

%% Model separation
% We make the assumption that two distinct models can be used to explain
% the output: one with "constant", high value of y; and another model.
% We learn a classifier to separate the two.
% The threshold was chosen from observation of the output data.

[betaClassifier, tX1, y1, tX2, y2] = separateDataSet(y, tX, 6200);

% We now easily spot some outliers in the second model
outliers2 = y2 > 7200;
tX2 = tX2(~outliers2, :);
y2 = y2(~outliers2);

%% Learn model M2
% As observed, the second model is simply a constant

betaM2 = zeros(size(tX2, 2), 1);
betaM2(1) = mean(y2);

%% Learn model M1
% On the other hand, the first model is not obvious and should be learnt
% using a ML technique.

% Basis function expansion
% It allows us to fit a more complex model, but might produce overfitting.
% This is why we use ridge regression.
% We found that polynomials of degree 4 produced the best tradeoff.
% Warning: do not apply basis extension to binary variables!
% Note that this transformation must be applied to all new input data at
% the time of prediction.
tX1 = [ones(size(tX1, 1), 1) createPoly(tX1(:, 2:36), 4) tX1(:, 37:end)];

k = 5; % k-fold cross validation
lambdas = logspace(0, 2, 100);
betaM1 = ridgeRegressionAuto(y1, tX1, k, lambdas);

%% Constitute the best predictor
% Two steps: classify to select the model, then apply the necessary transformation
% and finaly compute the prediction.

predictor = @(tX) hybridPredictor(tX, betaClassifier, betaM1, betaM2);

trErrHybrid = computeRmse(y, predictor(tX));
teErrHybrid = computeRmse(y_test, predictor(tX_test));
fprintf('Error with the hybrid predictor: %f | %f\n', trErrHybrid, teErrHybrid);

%% Predict test data using the best model

yPredicted = predictor(tXtoPredict);

path = './data/regression-output.csv';
writeCsv(yPredicted, path);
disp(['Predictions output to ', path]);
