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

selector = @(x) abs(x) < 0.1;
[correlatedVariables, correlations] = findCorrelations(selector, X, y);
correlatedVariables

% X26 and X35 are the features with the strongest correlations

%% Regression fitting only on X26 and X35

Xt = [X(:,26) X(:,35)];

%% Comparison between the above Xl and the full X feature matrices

seedsNb = 50;
res = compareFeaturesSet(y, X, Xt, seedsNb);

%% Regression fit selecting only one out of 2 of the strongly correlated variables
% Removed variables (highest correlation between ft and lowest to output)
% 13.0000
% 14.0000
% 16.0000
% 18.0000
% 25.0000
% 27.0000
% 29.0000

Xl = [X(:,1:12) X(:,15) X(:,17) X(:,19:24) X(:,26) X(:,28) X(:,30:end)];

%% Comparison between the above Xl and the full X feature matrices

seedsNb = 50;
res = compareFeaturesSet(y, X, Xl, seedsNb);

% Results with Xl are very close to the one we have with X and we have the
% same stability of results