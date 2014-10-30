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