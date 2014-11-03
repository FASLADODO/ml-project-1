% Regression fitting
addpath(genpath('./data'), genpath('../data'));
addpath(genpath('./src'), genpath('../src'));

clear;
load('regression.mat');

X = X_train;
y = normalized(y_train);
% Our goal is to predict values for this input
XtoPredict = X_test;

% This seed is used to reset the RNG when needed to obtain comparable
% splits (e.g. when trying to select lambda)
seed = randi(1000);

% Prediction function for any linear model
predictLinear = @(tX, beta) tX * beta;
% Compute RMSE error for a given prediction
computeError = @computeRmse;
% We'll perform k-fold cross validation to estimate error for each method
K = 10; 

% Preprocessing
% We perform dummy variables encoding on categorical variables only
categoricalVariables = [36 38 40 43 44];
X = dummyEncoding(X, categoricalVariables);
XtoPredict = dummyEncoding(XtoPredict, categoricalVariables);

% Normalize features (except the discrete ones)
[X(:,1:35), XtoPredict(:,1:35)] = normalized(X(:,1:35), XtoPredict(:,1:35));

N = length(y);
tX = [ones(N, 1) X];
tXtoPredict = [ones(size(XtoPredict, 1), 1) XtoPredict];

%% Get a baseline for the cost by fitting a one-variable model
learnConstant = @(y, tX) [mean(y); zeros(size(tX, 2) - 1, 1)];
[trErr0, teErr0] = kFoldCrossValidation(y, tX, K, learnConstant, predictLinear, computeError);
fprintf('Estimated error with a 1-parameter model: %f | %f\n', trErr0, teErr0);

%% Train a linear model using simple least squares
[trErrLS, teErrLS] = kFoldCrossValidation(y, tX, K, @leastSquares, predictLinear, computeError);
fprintf('Estimated error with least squares: %f | %f\n', trErrLS, teErrLS);

%% Train a linear model using simple least squares with gradient descent
alpha = 0.1; % Step size
learnLeastSquares = @(y, tX) leastSquaresGD(y, tX, alpha);
[trErrLSGD, teErrLSGD] = kFoldCrossValidation(y, tX, K, learnLeastSquares, predictLinear, computeError);
fprintf('Estimated error with least squares (gradient descent): %f | %f\n', trErrLSGD, teErrLSGD);

%% Train a linear model using ridge regression
% Note that ridge regression uses its own cross validation to select lambda
lambdas = logspace(0, 2, 100);
learnRidgeRegression = @(y, tX) ridgeRegressionAuto(y, tX, 5, lambdas, seed);
[trErrRR, teErrRR] = kFoldCrossValidation(y, tX, K, learnRidgeRegression, predictLinear, computeError);
fprintf('Estimated error with ridge regression: %f | %f\n', trErrRR, teErrRR);

%% Train a linear model using basis extension and ridge regression
% Polynomial basis extension (only the real-valued variables)
tXExtended = [ones(size(tX, 1), 1) createPoly(tX(:, 2:36), 4) tX(:, 37:end)];
[trErrPRR, teErrPRR] = kFoldCrossValidation(y, tXExtended, K, learnRidgeRegression, predictLinear, computeError);
fprintf('Estimated error with polynomial basis extension and ridge regression : %f | %f\n', trErrPRR, teErrPRR);

%% Constitute the best predictor and estimate its test error
% The threshold was chosen from observation of the output data
threshold = 2; % For a non-normalized output: 6200
learn = @(y, tX) learnHybridModel(y, tX, threshold);
predict = @(tX, betas) hybridPredictor(tX, betas{1}, betas{2}, betas{3});

[trErrHybrid, teErrHybrid] = kFoldCrossValidation(y, tX, K, learn, predict, @computeRmse);
fprintf('Error with the hybrid predictor: %f | %f\n', trErrHybrid, teErrHybrid);

%% Predict test data using the best predictor
% Now that we're confident about the validity of our approach, we can use
% the whole training dataset to learn the best possible classifier
bestBeta = learn(y, tX);
yHat = predict(tXtoPredict, bestBeta);

path = './results/predictions_regression.csv';
writeCsv(yHat, path);
disp(['Predictions output to ', path]);

%% Output error estimates
path = './results/test_errors_regression.csv';
fid = fopen(path, 'w');
fprintf(fid, 'method,rmse\n');
fprintf(fid, 'leastSquaresGD,%.3f\n', teErrLSGD);
fprintf(fid, 'leastSquares,%.3f\n', teErrLS);
fprintf(fid, 'ridgeRegression,%.3f\n', teErrPRR);
fprintf(fid, 'hybridModel,%.3f\n', teErrHybrid);
fclose(fid);

disp(['Error estimates output to ', path]);