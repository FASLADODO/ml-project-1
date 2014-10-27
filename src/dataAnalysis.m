addpath('../data');

%% Regression data exploratory analysis
clear;
load('regression.mat');

X = X_train;
y = y_train;

% We have N = 1400, D = 44
size(X);
size(y);

% Normalize the features except discrete ones
X(:,1:35) = normalized(X(:,1:35));

%% Output Visualization

hist(y);

% there is one main gaussian + one other thing that we cannot understand :
% too much data to be outliers : a second gaussian maybe ?


%% Input Visualization
figure;
side = 7;
for k = 1:size(X, 2)
    subplot(side, side, k);
    hist(X(:, k));
    title(['X', int2str(k)]);
end;

% X35 looks weird : outliers or 2 gaussians ? 
% Comparing input to ouput reveals we might face two models superposed. We
% might be able to separate them based on the categorical data or thanks to
% X35. Data looks linearly separable -> classification and then regression
% ??

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

%% First try at separating the two models

t = y>6000;
figure;
side = 7;
for k = 1:size(X, 2)
    subplot(side, side, k);
    plot(X(t, k), y(t), '.r');
    hold on;
    plot(X(~t, k), y(~t), '.b');
    title(['X', int2str(k)]);
end;

% Interesting plot to put in the report to show the categorical variables
% are not correlated with the hypothetical two models 

%% Compute the correlation between the features and spot the largest ones

% Eliminate the duplicate features

% Detect and delete the outliers
