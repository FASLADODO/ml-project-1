% Regression fitting
addpath(genpath('./data'), genpath('../data'));
addpath(genpath('./src'), genpath('../src'));

clear;
load('regression.mat');

X = X_train;
y = y_train;

% Prediction function for any linear model
predictLinear = @(tX, beta) tX * beta;
% Compute RMSE error for a given prediction
computeError = @computeRmse;
% We'll perform k-fold cross validation to estimate error for each method
K = 10; 

%% Preprocessing
% We perform dummy variables encoding on categorical variables only
categoricalVariables = [36 38 40 43 44];
X = dummyEncoding(X, categoricalVariables);

% Normalize features (except the discrete ones)
X(:,1:35) = normalized(X(:,1:35));

N = length(y);
tX = [ones(N, 1) X];

%% Get a baseline for the cost by fitting a one-variable model
learnConstant = @(y, tX) [mean(y); zeros(size(tX, 2) - 1, 1)];
[trErr0, teErr0] = kFoldCrossValidation(y, tX, K, learnConstant, predictLinear, computeError);
fprintf('Estimated error with a 1-parameter model: %f | %f\n', trErr0, teErr0);

%% Train a linear model using simple least squares
[trErrLS, teErrLS] = kFoldCrossValidation(y, tX, K, @leastSquares, predictLinear, computeError);
fprintf('Estimated error with least squares: %f | %f\n', trErrLS, teErrLS);

%% Train a linear model using ridge regression
% Note that ridge regression uses its own cross validation to select lambda
lambdas = logspace(0, 2, 100);
learnRidgeRegression = @(y, tX) ridgeRegressionAuto(y, tX, 10, lambdas);
[trErrRR, teErrRR] = kFoldCrossValidation(y, tX, K, learnRidgeRegression, predictLinear, computeError);
fprintf('Estimated error with ridge regression: %f | %f\n', trErrRR, teErrRR);

%% Train a linear model using basis extension and ridge regression
% TODO

%% Predict test data using the best model
% @see regressionHybridFit.m
