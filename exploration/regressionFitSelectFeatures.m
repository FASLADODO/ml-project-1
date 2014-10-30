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

%% Compute the correlation between the features and spot the largest ones
 
selector = @(x) abs(x) > 0.4;
[correlatedVariables, correlations] = findCorrelations(selector, X);
correlatedVariables

figure;
imagesc(correlations);
colorbar;

%% Correlation input/output

selector = @(x) abs(x) > 0.4;
[correlatedVariables, correlations] = findCorrelations(selector, X, y);
correlatedVariables

% X26 and X35 are the features with the strongest correlations

%% 

Xt = [X(:,26) X(:,35)];

[X, y, X_test, y_test] = split(y, Xt, 0.8, 1);

N = length(y);
tX = [ones(N, 1) X];
tX_test = [ones(length(y_test), 1) X_test];