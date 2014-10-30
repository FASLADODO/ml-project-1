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
X = [X(:,26) X(:,35)];


%%
proportion = 0.8;

for degree = 1:7
    
    % get train and test data
    [XTr, yTr, XTe, yTe] = split(y,X,proportion,1);

	% form tX
	tXTr = [ones(length(yTr), 1) createPoly(XTr, degree)];
	tXTe = [ones(length(yTe), 1) createPoly(XTe, degree)];

	% least squares
	beta = leastSquares(yTr, tXTr);

	% train and test RMSE
	rmseTr =  computeRmse(yTr, tXTr*beta); 
	rmseTe =  computeRmse(yTe, tXTe*beta);  

	% print 
	fprintf('Degree %.2f: Train RMSE :%0.4f Test RMSE :%0.4f\n', degree, rmseTr, rmseTe);
end