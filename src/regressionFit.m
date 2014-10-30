% Regression fitting
addpath(genpath('./data'), genpath('../data'));
addpath(genpath('./src'), genpath('../src'));

clear;
load('regression.mat');

% TODO: vary seed to confirm
[X, y, X_test, y_test] = split(y_train, X_train, 0.8, 1);

N = length(y);
tX = [ones(N, 1) X];
tX_test = [ones(length(y_test), 1) X_test];

% Normalize the features except discrete ones
X(:,1:35) = normalized(X(:,1:35));
X_test(:,1:35) = normalized(X_test(:,1:35));

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
proportion = 0.5; % Train / test split
k = 5; % k-fold cross validation
lambdas = logspace(-4, 4, 100);
% We leave X_test and y_test out of the learning process of ridge
% regression to be able to test its results on truly fresh data
betaRR = ridgeRegressionAuto(y, tX, proportion, k, lambdas);

trErrRR = computeRmse(y, tX * betaRR);
teErrRR = computeRmse(y_test, tX_test * betaRR);
fprintf('Error with ridge regression: %f | %f\n', trErrRR, teErrRR);