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

% Normalize features (except the discrete ones)
[X(:,1:29), X_te(:,1:29)] = normalized(X(:,1:29), X_te(:,1:29));

%% Removing the outliers
threshold = 10; % outliers are more than 10 standard deviation from the median
[X, y] = removeOutliers(X, y, threshold);


% prepare data
tX = [ones(size(X,1), 1) X];
tX_te = [ones(size(X_te,1), 1) X_te]; 
y(y < 1) = 0; % changing {-1, 1} to {0, 1}
alpha = 1e-03; % step size


K = 2; % CV folds
% split data in k fold (create indices only)
setSeed(1);
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
err = err / K




% predict using best model
beta_bestmodel = penLogisticRegressionAuto(y, tX, K);
pHatn = exp(logSigmoid(tX * beta_bestmodel)); % probability p(y=1|data)
