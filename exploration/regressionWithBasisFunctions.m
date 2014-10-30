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

% Normalize the features except discrete ones
X(:,1:35) = normalized(X(:,1:35));
%X = [X(:,26) X(:,35)];


%%
proportion = 0.8;

% TO DO : different seeds on same plot
for degree = 1:7
    
    % get train and test data
    [XTr, yTr, XTe, yTe] = split(y,X,proportion,1);

	% form tX
	tXTr = [ones(length(yTr), 1) createPoly(XTr, degree)];
	tXTe = [ones(length(yTe), 1) createPoly(XTe, degree)];

	% least squares
	%beta = leastSquares(yTr, tXTr);

    % ridge regression
    k = 5; % k-fold cross validation
    lambdas = logspace(-2, 4, 50);
    % We leave X_test and y_test out of the learning process of ridge
    % regression to be able to test its results on truly fresh data
    beta = ridgeRegressionAuto(yTr, tXTr, proportion, k, lambdas);
    
	% train and test RMSE
	rmseTr =  computeRmse(yTr, tXTr*beta); 
	rmseTe =  computeRmse(yTe, tXTe*beta);  

	% print 
	fprintf('Degree %.2f: Train RMSE :%0.4f Test RMSE :%0.4f\n', degree, rmseTr, rmseTe);
end