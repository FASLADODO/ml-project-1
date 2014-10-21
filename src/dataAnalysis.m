addpath('../data');

%% Regression data exploratory analysis
clear;
load('regression.mat');

% We have N = 1400, D = 44
size(X_train);
size(y_train);

% Normalize the features
X = normalized(X_train);

% Plotting the features individually against Y
figure;
offset = 10;
side = 7;
for k = 1:size(X, 2)
    subplot(side, side, k);
    plot(X(:, k), y_train, '.');
    title(['X', int2str(k), ' versus Y']);
end;
% Note that features 37 to 44 have discrete values!

% Compute the correlation between the features and spot the largest ones

% Eliminate the duplicate features

% Detect and delete the outliers
