addpath(genpath('./data'), genpath('../data'));
addpath(genpath('./src'), genpath('../src'));

%% Data pre-processing
clear;
load('classification.mat');

X = X_train;
y = y_train;
X_te = X_test;


categoricalVariables = [1 15 30];
X = dummyEncoding(X, categoricalVariables);
X_te = dummyEncoding(X_te, categoricalVariables);

% Basis extension
degree = 0.5;
X = [X(:, 1:29) abs(X(:, 1:29)).^degree X(:, 30: end)];
X_te = [X_te(:, 1:29) abs(X_te(:, 1:29)).^degree X_te(:, 30: end)];

% Normalize features (except the discrete ones)
[X(:,1:29), X_te(:,1:29)] = normalized(X(:,1:29), X_te(:,1:29));

%% Removing the outliers
threshold = 10; % outliers are more than 10 standard deviation from the median
[X, y] = removeOutliers(X, y, threshold);

% prepare data
tX = [ones(size(X,1), 1) X];
tX_te = [ones(size(X_te,1), 1) X_te];

y(y < 1) = 0; % changing {-1, 1} to {0, 1}
alpha = 0.5; % step size

%% Estimate train and test error with k-fold cross validation
K = 5; % CV folds
% split data in k fold (create indices only)
N = size(y,1);
idx = randperm(N);
Nk = floor(N/K);
for k = 1:K
    idxCV(k,:) = idx(1+(k-1)*Nk:k*Nk);
end

% k-fold cross-validation
err = zeros(4, 6);
for k = 1:K
    % get k'th subgroup in test, others in train
    idxTe = idxCV(k,:);
    idxTr = idxCV([1:k-1 k+1:end],:);
    idxTr = idxTr(:);
    yTe = y(idxTe);
    XTe = tX(idxTe,:);
    yTr = y(idxTr);
    XTr = tX(idxTr,:);

    beta0 = mean(yTr); 
    betaLS = leastSquares(yTr, XTr);
    betaLR = logisticRegression(yTr, XTr, alpha);
    betaPLR = penLogisticRegressionAuto(yTr, XTr, K);
    err = err + [computeErrorEstimate(XTr(:, 1), yTr, beta0) computeErrorEstimate(XTe(:, 1), yTe, beta0); ...
    			computeErrorEstimate(XTr, yTr, betaLS) computeErrorEstimate(XTe, yTe, betaLS); ...
    			computeErrorEstimate(XTr, yTr, betaLR) computeErrorEstimate(XTe, yTe, betaLR); ...
    			computeErrorEstimate(XTr, yTr, betaPLR) computeErrorEstimate(XTe, yTe, betaPLR)];
end


% 								TrainErr						TestErr
%					 ---------------------------------------------------------------
% 						RMSE 	0-1loss 	logLoss  |	RMSE 	0-1loss 	logLoss
%					 ---------------------------------------------------------------
% 1-parameter model |	
% least squares 	|
% LR 				|
% PLR 				|
err = err / K;
err

%% Output prediction
bestBeta = logisticRegression(y, tX, alpha);
% probability p(y=1|data)
[~, pHat] = binaryPrediction(tX_te, bestBeta);
path = './results/predictions_classification.csv';
csvwrite(path, pHat);
disp(['Predictions output to ', path]);

% Output error estimates
path = './results/test_errors_classification.csv';
fid = fopen(path, 'w');
fprintf(fid, 'method,rmse,0-1-loss,log-loss\n');
fprintf(fid, 'logisticRegression,%.3f,%.3f,%.3f\n', err(3,4), err(3,5), err(3,6));
fprintf(fid, 'penLogisticRegression,%.3f,%.3f,%.3f\n', err(4,4), err(4,5), err(4,6));
fclose(fid);
disp(['Error estimates output to ', path]);