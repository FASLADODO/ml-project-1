% Classification fitting
addpath(genpath('./data'), genpath('../data'));
addpath(genpath('./src'), genpath('../src'));

clear;
load('classification.mat');
XtoPredict = X_test;

% Relabel -1 to 1 in the output
y_train(y_train < 1) = 0;

% TODO: vary seed to confirm the stability of the results
[X, y, X_test, y_test] = split(y_train, X_train, 0.8, 1);

%% Preprocessing
% We perform dummy variables encoding on categorical variables only
categoricalVariables = [1 15 30];
X = dummyEncoding(X, categoricalVariables);
X_test = dummyEncoding(X_test, categoricalVariables);
XtoPredict = dummyEncoding(XtoPredict, categoricalVariables);

% Normalize features (except the discrete ones)
notNormalized = X(:,1:29);
[X(:,1:29), X_test(:,1:29)] = normalized(notNormalized, X_test(:,1:29));
[~, XtoPredict(:,1:29)] = normalized(notNormalized, XtoPredict(:,1:29));
clear notNormalized;

% Removing the outliers
threshold = 10; % outliers are more than 10 standard deviation from the median
[X, y] = removeOutliers(X, y, threshold);

% Basis function expansion
% We found that polynomials of degree ??? produced the best tradeoff.
tX = [ones(length(y), 1) X];
tX_test = [ones(length(y_test), 1) X_test];
% TODO
%tX = [ones(length(y), 1) createPoly(X(:, 1:29), 4) X(:, 30:end)];
%tX_test = [ones(length(y_test), 1) createPoly(X_test(:, 1:29), 4) X_test(:, 30:end)];

%% Get a baseline for the cost by fitting a one-variable model
beta0 = mean(y);
trErr0 = computeLogisticRegressionMse(y, tX(:, 1), beta0);
teErr0 = computeLogisticRegressionMse(y_test, tX_test(:, 1), beta0);
fprintf('Base error with a 1-parameter model: %f | %f\n', trErr0, teErr0);

%% Train a linear model using simple least squares
betaLS = leastSquares(y, tX);

trErrLS = computeLogisticRegressionMse(y, tX, betaLS);
teErrLS = computeLogisticRegressionMse(y_test, tX_test, betaLS);
fprintf('Error with least squares: %f | %f\n', trErrLS, teErrLS);

%% Train a linear model using logistic regression
alpha = 0.5; % Step size
betaLR = logisticRegression(y, tX, alpha);

trErrLR = computeLogisticRegressionMse(y, tX, betaLR);
teErrLR = computeLogisticRegressionMse(y_test, tX_test, betaLR);
fprintf('Error with logistic regression: %f | %f\n', trErrLR, teErrLR);

%% Train a linear model using penalized logistic regression
k = 5; % k-fold cross validation
lambdas = logspace(0, 4, 50);
betaPLR = penLogisticRegressionAuto(y, tX, k, lambdas);

trErrPLR = computeLogisticRegressionMse(y, tX, betaPLR);
teErrPLR = computeLogisticRegressionMse(y_test, tX_test, betaPLR);
fprintf('Error with penalized logistic regression: %f | %f\n', trErrPLR, teErrPLR);

%% Predict test data using the best model
% TODO

% TODO: don't forget to relabel 0 to -1 in the output

% Export predictions to CSV