addpath(genpath('./data'), genpath('../data'));
addpath(genpath('./src'), genpath('../src'));

%% Data pre-processing
clear;
load('regression.mat');

X = X_train;
y = y_train;
N = length(y);

% We have N = 1400, D = 44
size(X);
size(y);

% We ignore the categorical features in this analysis
X = normalized(X(:,1:35));
%X = [X(:,1:12) X(:,15) X(:,17) X(:,19:24) X(:,26) X(:,28) X(:,30:end)];
% results seems nice but ridge reg graphes are little bit less impressives
%X = [X(:,26) X(:,35)];


%% Polynomials tests with ridge regression
% We apply here ridge regression to test our model using polynomials
% functions because the resulting matrix of the applied basis functions is
% singular

proportion = 0.8;
maxDegree = 6;
maxSeeds = 50;

rmseTr = zeros(maxSeeds,maxDegree);
rmseTe = zeros(maxSeeds,maxDegree);
rmseTrMean = zeros(maxDegree);
rmseTeMean = zeros(maxDegree);

degrees = [1:maxDegree];
for degree = 1:length(degrees);
    figure;
    for s = 1:maxSeeds % # of seeds
        % get train and test data with given seed and proportion
        [XTr, yTr, XTe, yTe] = split(y,X,proportion,s);

        % form tX
        tXTr = [ones(length(yTr), 1) createPoly(XTr, degree)];
        tXTe = [ones(length(yTe), 1) createPoly(XTe, degree)];

        % ridge regression
        k = 5; % k-fold cross validation
        lambdas = logspace(-1, 4, 50);
        % We leave X_test and y_test out of the learning process of ridge
        % regression to be able to test its results on truly fresh data
        [beta, trainingErr, testErr] = ridgeRegressionAuto(yTr, tXTr, proportion, k, lambdas);
      
        ridgeTrErr(:,s) = trainingErr;
        ridgeTeErr(:,s) = testErr;
        
        % train and test RMSE
        rmseTr(s,degree) =  computeRmse(yTr, tXTr*beta); 
        rmseTe(s,degree) =  computeRmse(yTe, tXTe*beta);  
        
        % print for each seeds
        %fprintf('Degree %.2f: Train RMSE :%0.4f Test RMSE :%0.4f\n', degree, rmseTr(s,degree), rmseTe(s,degree));
    end
    
    rmseTrMean(degree) = mean(rmseTr(:,degree));
    rmseTeMean(degree) = mean(rmseTe(:,degree));
    rmseTrStd(degree) = std(rmseTr(:,degree));
    rmseTeStd(degree) = std(rmseTe(:,degree));
    fprintf('Degree %d with %d seeds - Train RMSE: %0.4f (std: %0.4f) Test RMSE: %0.4f (std: %0.4f)\n', degree, maxSeeds, rmseTrMean(degree), rmseTrStd(degree), rmseTeMean(degree), rmseTeStd(degree));

    % plot training and test error wrt lambdas averaged on different seeds
    % plot on 2 different figures for the report
    rigdeTrErrMean = mean(ridgeTrErr,2);
    rigdeTeErrMean = mean(ridgeTeErr,2);
    semilogx(lambdas, rigdeTrErrMean, '.-b');
    hold on;
    semilogx(lambdas, rigdeTeErrMean, '.-r');
    xlabel('Lambda');
    ylabel('Training (blue) and test (red) error');
    title(sprintf('Ridge regression with polynomial degree %d',degree))
end

%% Boxplots on different polynomials to visualize above printed results
xLabel = 'Degree of polynomial basis extension';
yLabel = ['RMSE over ', int2str(maxSeeds), ' seeds'];

figure;
boxplot(rmseTr, 'notch', 'on');
title('RMSE (train data) using polynomials basis extension');
savePlot('./report/figures/basis-extension-train-error.pdf', xLabel, yLabel);

figure;
boxplot(rmseTe, 'notch', 'on');
title('RMSE (test data) using polynomials basis extension')
savePlot('./report/figures/basis-extension-test-error.pdf', xLabel, yLabel);

% Polynomial with degree 4 seems the best (almost smallest errors + a very
% reasonable variance over repetition on different seeds = seems stable)
% TODO: validation with CV