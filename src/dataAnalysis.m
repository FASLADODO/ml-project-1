addpath('../data');

%% Regression data exploratory analysis
clear;
load('regression.mat');

% We have N = 1400, D = 44
size(X_train);
size(y_train);

% Normalize the features
X = normalized(X_train);

%% Plotting the features individually against Y
figure;
side = 7;
for k = 1:size(X, 2)
    subplot(side, side, k);
    plot(X(:, k), y_train, '.');
    title(['X', int2str(k), ' versus Y']);
end;
% Note that features 37 to 44 have discrete values!

%% Plotting the features against each other
figure;
offset = 0;
side = 10;
for i = 1:side
    for j  = 1:side
        subplot(side, side, (i - 1) * side + j);
        plot(X(:, i+offset), X(:, j+offset), '.');
        title(['X', int2str(i+offset), ' versus X', int2str(j+offset)]);
    end;
end;

% We spot some correlations (but not that many).
% Use ACP for dimensionality reduction?


%% Compute the correlation between the features and spot the largest ones

% Eliminate the duplicate features

% Detect and delete the outliers
