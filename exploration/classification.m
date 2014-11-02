addpath(genpath('./data'), genpath('../data'));
addpath(genpath('./src'), genpath('../src'));

%% Data pre-processing
clear;
load('classification.mat');

X = X_train;
y = y_train;
X_te = X_test;


% categorical variables : X1, X15, X30 -> move it at the end of the X
% matrix : now X30, X31, X32 are categorical variables
X = [X(:,2:14) X(:,16:29) X(:,31:end) X(:,1) X(:,15) X(:,30)];
X(:,1:29) = normalized(X(:,1:29));

X_te = [X_te(:,2:14) X_te(:,16:29) X_te(:,31:end) X_te(:,1) X_te(:,15) X_te(:,30)];
X_te(:,1:29) = normalized(X_te(:,1:29));


% dummy encoding TODO : debug and use the dummyEncoding function
dummy0 = dummyvar(X(:,30));
dummy1 = dummyvar(X(:,31));
dummy2 = dummyvar(X(:,32));
X = [X(:,1:29) dummy0(:, 1:end-1) dummy1(:, 1:end-1) dummy2(:, 1:end-1)]; % using only k-1 dummy variables for categorical variables with k categories


dummy0_te = dummyvar(X_te(:,30));
dummy1_te = dummyvar(X_te(:,31));
dummy2_te = dummyvar(X_te(:,32));
X_te = [X_te(:,1:29) dummy0_te(:, 1:end-1) dummy1_te(:, 1:end-1) dummy2_te(:, 1:end-1)]; % using only k-1 dummy variables for categorical variables with k categories



%% Removing the outliers
threshold = 10; % outliers are more than 10 standard deviation from the median
[X, y] = removeOutliers(X, y, threshold);


% prepare data
tX = [ones(size(X,1), 1) X];
tX_te = [ones(size(X_te,1), 1) X_te]; 
y(y<1) = 0; % changing {-1, 1} to {0, 1}
alpha = 1e-03; % step size



K = 10; % CV folds
% split data in k fold (create indices only)
setSeed(1);
N = size(y,1);
idx = randperm(N);
Nk = floor(N/K);
for k = 1:K
    idxCV(k,:) = idx(1+(k-1)*Nk:k*Nk);
end



%% Logistic Regression
% Prediction
beta_lr = logisticRegression(y, tX, alpha);
pHatn = 1.0 ./ (1.0 + exp(-tX_te * beta_lr)); % probability p(y=1|data)

% Estimate of test error using k-fold cross validation
% k-fold cross-validation
loss = [];
for k = 1:K
    % get k'th subgroup in test, others in train
    idxTe = idxCV(k,:);
    idxTr = idxCV([1:k-1 k+1:end],:);
    idxTr = idxTr(:);
    yTe = y(idxTe);
    XTe = tX(idxTe,:);
    yTr = y(idxTr);
    XTr = tX(idxTr,:);

    % train beta on training data 
    beta = logisticRegression(yTr, XTr, alpha);

    % compute different kinds of losses : RMSE, 0-1 loss and logLoss(entropy)
    [RMSE, zero_one, logLoss] = computeErrorEstimate(XTe, yTe, beta);
    loss = [loss ; [RMSE, zero_one, logLoss]];

end
 
mean(loss)
    


%% Penalized Logistic Regression
% Prediction
[beta_plr, trainingErr, testErr] = penLogisticRegressionAuto(y, tX);
pHatn = 1.0 ./ (1.0 + exp(-tX_te * beta_plr)); % probability p(y=1|data)

% Estimate of test error
% k-fold cross-validation
loss_pen = [];
for k = 1:K
    % get k'th subgroup in test, others in train
    idxTe = idxCV(k,:);
    idxTr = idxCV([1:k-1 k+1:end],:);
    idxTr = idxTr(:);
    yTe = y(idxTe);
    XTe = tX(idxTe,:);
    yTr = y(idxTr);
    XTr = tX(idxTr,:);

    % train beta on training data 
    [beta, trainingErr, testErr] = penLogisticRegressionAuto(yTr, XTr);

    % compute different kinds of losses : RMSE, 0-1 loss and logLoss(entropy)
    [RMSE, zero_one, logLoss] = computeErrorEstimate(XTe, yTe, beta);
    loss_pen = [loss_pen ; [RMSE, zero_one, logLoss]];

end
 
mean(loss_pen)



%TODO : really redundent code 





