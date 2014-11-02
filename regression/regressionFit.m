% Regression fitting
addpath(genpath('./data'), genpath('../data'));
addpath(genpath('./src'), genpath('../src'));

clear;
load('regression.mat');

% We work with a fixed seed to better understand improvements to our
% ML methods
[X, y, X_test, y_test] = split(y_train, X_train, 0.8, 1);

%% Preprocessing
% We perform dummy variables encoding on categorical variables only
categoricalVariables = [36 38 40 43 44];
X = dummyEncoding(X, categoricalVariables);
X_test = dummyEncoding(X_test, categoricalVariables);

% Normalize features (except the discrete ones)
[X(:,1:35), X_test(:,1:35)] = normalized(X(:,1:35), X_test(:,1:35));

N = length(y);
tX = [ones(N, 1) X];
tX_test = [ones(length(y_test), 1) X_test];

%% Get a baseline for the cost by fitting a one-variable model
beta0 = mean(y);
trErr0 = computeRmse(y, tX(:, 1) * beta0);
teErr0 = computeRmse(y_test, tX_test(:, 1) * beta0);
fprintf('Base error with a 1-parameter model: %f | %f\n', trErr0, teErr0);

%% Train a linear model using simple least squares
betaLS = leastSquares(y, tX);

trErrLS = computeRmse(y, tX * betaLS);
teErrLS = computeRmse(y_test, tX_test * betaLS);
fprintf('Error with least squares: %f | %f\n', trErrLS, teErrLS);

%% Train a linear model using ridge regression
k = 10; % k-fold cross validation
lambdas = logspace(0, 2, 100);
% We leave X_test and y_test out of the learning process of ridge
% regression to be able to test its results on truly fresh data
betaRR = ridgeRegressionAuto(y, tX, k, lambdas);

trErrRR = computeRmse(y, tX * betaRR);
teErrRR = computeRmse(y_test, tX_test * betaRR);
fprintf('Error with ridge regression: %f | %f\n', trErrRR, teErrRR);

%% Predict test data using the best model
% @see regressionHybridFit.m
