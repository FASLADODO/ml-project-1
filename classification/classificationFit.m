% Classification fitting
addpath(genpath('./data'), genpath('../data'));
addpath(genpath('./src'), genpath('../src'));

clear;
load('classification.mat');

% Relabel -1 to 0 in the output
y_train(y_train < 1) = 0;

X = X_train;
y = y_train;
XtoPredict = X_test;

% This seed is used to reset the RNG when needed to obtain comparable
% splits (e.g. when trying to select lambda)
seed = randi(1000);

% Prediction function for any linear model
predictLinear = @(tX, beta) tX * beta;
% Compute RMSE error for a given prediction
% TODO: support all kind of loss functions
computeError = @computeLRMse;
% We'll perform k-fold cross validation to estimate error for each method
K = 10;

% Preprocessing
% We perform dummy variables encoding on categorical variables only
categoricalVariables = [1 15 30];
X = dummyEncoding(X, categoricalVariables);
XtoPredict = dummyEncoding(XtoPredict, categoricalVariables);

% Normalize features (except the discrete ones)
[X(:,1:29), XtoPredict(:,1:29)] = normalized(X(:,1:29), XtoPredict(:,1:29));

% Removing the outliers
deviations = 10; % Outliers are more than 10 standard deviation from the median
[X, y] = removeOutliers(X, y, deviations);

tX = [ones(length(y), 1) X];
tXtoPredict = [ones(size(XtoPredict, 1), 1) XtoPredict];

%% Get a baseline for the cost by fitting a one-variable model
learnConstant = @(y, tX) [mean(y); zeros(size(tX, 2) - 1, 1)];
[trErr0, teErr0] = kFoldCrossValidation(y, tX, K, learnConstant, predictLinear, computeError);
fprintf('Estimated error with a 1-parameter model: %f | %f\n', trErr0, teErr0);

%% Train a linear model using simple least squares
[trErrLS, teErrLS] = kFoldCrossValidation(y, tX, K, @leastSquares, predictLinear, computeError);
fprintf('Estimated error with least squares: %f | %f\n', trErrLS, teErrLS);

%% Train a linear model using logistic regression
alpha = 0.5; % Step size
learnLogReg = @(y, tX) logisticRegression(y, tX, alpha);
[trErrLR, teErrLR] = kFoldCrossValidation(y, tX, K, learnLogReg, predictLinear, computeError);
fprintf('Estimated error with logistic regression: %f | %f\n', trErrLR, teErrLR);

%% Train a linear model using penalized logistic regression
% Note penalized logistic regression uses its own kCV to select lambda
lambdas = logspace(0, 1, 20);
learnPenLogReg = @(y, tX) penLogisticRegressionAuto(y, tX, 5, lambdas, seed);
[trErrPLR, teErrPLR] = kFoldCrossValidation(y, tX, K, learnPenLogReg, predictLinear, computeError);
fprintf('Estimated error with penalized logistic regression: %f | %f\n', trErrPLR, teErrPLR);

%% Train a linear model using basis extension and penalized logistic regression
lambdas = logspace(-1, 5, 20);
learnPenLogReg = @(y, tX) penLogisticRegressionAuto(y, tX, 5, lambdas, seed);

% TODO: find best basis extension
degree = 2;
%tXExtended = [ones(length(y), 1) createPoly(tX(:, 2:30), degree) tX(:, 31:end)];
%tXExtended = [ones(length(y), 1) tX(:, 2:30) tX(:, 2:30).^0.5 tX(:, 31:end)];
%[trErrPLRE, teErrPLRE] = kFoldCrossValidation(y, tXExtended, K, learnPenLogReg, predictLinear, computeError);
%fprintf('Estimated error with basis extension and penalized logistic regression: %f | %f\n', trErrPLRE, teErrPLRE);

%% Predict test data using the best model
% TODO: actually use the best technique
bestBeta = learnPenLogReg(y, tX);
[yHat, pHat] = binaryPrediction(tXtoPredict, bestBeta);

% Export predictions to CSV
path = './results/classification-output.csv';
writeCsv(pHat, path);
disp(['Predictions output to ', path]);
