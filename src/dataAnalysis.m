addpath('../data');

%% Regression data exploratory analysis
clear;
load('regression.mat');

% We have N = 1400, D = 44
size(X_train)
size(y_train)

% Normalize the features

% Plotting the first ten features show a lot of redundancy in the data
plot(y_train, X_train(:, 1:10), '.');

% Compute the correlation between the features and spot the largest ones

% Eliminate the duplicate features

% Detect and delete the outliers