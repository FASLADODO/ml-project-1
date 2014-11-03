% Regression fitting
addpath(genpath('./data'), genpath('../data'));
addpath(genpath('./src'), genpath('../src'));

clear;
load('regression.mat');

X = X_train;
y = normalized(y_train);
% Our goal is to predict values for this input
XtoPredict = X_test;

% Compute RMSE error for a given prediction
computeError = @computeRmse;
% We'll perform k-fold cross validation to estimate error for each method
K = 10;

%% Preprocessing

% Normalize features (except the discrete ones)
[X(:,1:35), XtoPredict(:,1:35)] = normalized(X(:,1:35), XtoPredict(:,1:35));

% We perform dummy variables encoding on categorical variables only
categoricalVariables = [36 38 40 43 44];
X = dummyEncoding(X, categoricalVariables);
XtoPredict = dummyEncoding(XtoPredict, categoricalVariables);

N = length(y);
tX = [ones(N, 1) X];
tXtoPredict = [ones(size(XtoPredict, 1), 1) XtoPredict];

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
