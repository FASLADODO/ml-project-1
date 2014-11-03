addpath(genpath('./data'), genpath('../data'));
addpath(genpath('./src'), genpath('../src'));

clear;
load('classification.mat');

% Relabel -1 to 0 in the output
y_train(y_train < 1) = 0;

X = X_train;
y = y_train;

% We ignore the categorical features in this analysis
continuousVariables = true(size(X, 2), 1);
continuousVariables([1 15 30]) = false();
X = normalized(X(:, continuousVariables));

%% Polynomials tests with logistic regression

proportion = 0.8;
maxDegree = 6;
maxSeeds = 5;

rmseTr = zeros(maxSeeds,maxDegree);
rmseTe = zeros(maxSeeds,maxDegree);
rmseTrMean = zeros(maxDegree);
rmseTeMean = zeros(maxDegree);

degrees = 1:maxDegree;
for degree = 1:length(degrees);
    figure;
    for s = 1:maxSeeds % # of trials
        % get train and test data with given seed and proportion
        [XTr, yTr, XTe, yTe] = split(y, X, proportion, s);

        % form tX
        tXTr = [ones(length(yTr), 1) createPoly(XTr, degree)];
        tXTe = [ones(length(yTe), 1) createPoly(XTe, degree)];

        % penalized logistic regression
        k = 3;
        lambdaValues = logspace(0, 5, 50);
        [beta, trainingErr, testErr] = penLogisticRegressionAuto(yTr, tXTr, k, lambdaValues);
        
        logRegTrErr(:,s) = trainingErr;
        logRegTeErr(:,s) = testErr;
        
        % train and test RMSE
        rmseTr(s,degree) =  computeRmse(yTr, tXTr * beta); 
        rmseTe(s,degree) =  computeRmse(yTe, tXTe * beta);  
        
        % print for each seeds
        fprintf('Degree %.2f: Train RMSE: %0.4f Test RMSE: %0.4f\n', degree, rmseTr(s,degree), rmseTe(s,degree));
    end
    
    rmseTrMean(degree) = mean(rmseTr(:,degree));
    rmseTeMean(degree) = mean(rmseTe(:,degree));
    rmseTrStd(degree) = std(rmseTr(:,degree));
    rmseTeStd(degree) = std(rmseTe(:,degree));
    fprintf('Degree %d with %d seeds - Train RMSE: %0.4f (std: %0.4f) Test RMSE: %0.4f (std: %0.4f)\n', degree, maxSeeds, rmseTrMean(degree), rmseTrStd(degree), rmseTeMean(degree), rmseTeStd(degree));

    % plot training and test error wrt lambdas averaged on different seeds
    % plot on 2 different figures for the report
    logRegTrErrMean = mean(logRegTrErr,2);
    logRegTeErrMean = mean(logRegTeErr,2);
    semilogx(lambdaValues, logRegTrErrMean, '.-b');
    hold on;
    semilogx(lambdaValues, logRegTeErrMean, '.-r');
    xlabel('Lambda');
    ylabel('Training (blue) and test (red) error');
    title(sprintf('Penalized logistic regression with polynomial degree %d',degree))
end

%% Boxplots on different polynomials to visualize above printed results
xLabel = 'Degree of polynomial basis extension';
yLabel = ['RMSE over ', int2str(maxSeeds), ' seeds'];

figure;
boxplot(rmseTr, 'notch', 'on');
title('RMSE (train data) using polynomials basis extension');
savePlot('./report/figures/classification/basis-extension-train-error.pdf', xLabel, yLabel);

figure;
boxplot(rmseTe, 'notch', 'on');
title('RMSE (test data) using polynomials basis extension')
savePlot('./report/figures/classification/basis-extension-test-error.pdf', xLabel, yLabel);
