addpath('../data');

%% Regression data exploratory analysis
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

%% Output Visualization

hist(y);

% there is one main gaussian + one other thing that we cannot understand:
% too much data to be outliers: a second gaussian maybe ?


%% Input features visualization
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
% Warning, X41 is extremely skewed towards one class, it may be useless.

%% Plotting the features individually against Y
figure;
side = 7;
for k = 1:size(X, 2)
    subplot(side, side, k);
    plot(X(:, k), y_train, '.');
    title(['X', int2str(k), ' versus Y']);
end;
% There is a certain number of datapoints with high Y (output) value.
% This seems uncorrelated to all features except X35, which may enable us
% to separate these points.
% X19 and X26 look linearly correlated to Y.
% Note that features 37 to 44 have discrete values.

%% First try at separating the two models
% We suppose our data indeed follows two different models:
% 1. If X35 > some value, y = large constant => one parameter model
% 2. Else, y follows another, more complex model

t = y > 6000;
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

t = X(:, 35) > 1;
figure;
for k = 1:size(X, 2)
    subplot(side, side, k);
    plot(X(t, k), y(t), '.r');
    hold on;
    plot(X(~t, k), y(~t), '.b');
    title(['X', int2str(k)]);
end;

% Indeed, separating with respect to a threshold on X35 gives good results.
% Could we learn that threshold automatically to minimize error? (just a
% grid search on the threshold value?)

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

correlations = corr(X);

figure;
imagesc(correlations);
colorbar;

[corrI, corrJ] = find(abs(correlations) > 0.4);
idx = (corrI - corrJ > 0);
correlatedVariables = [corrI(idx) corrJ(idx)];
for i = 1:length(correlatedVariables)
    correlatedVariables(i, 3) = correlations(correlatedVariables(i, 1), correlatedVariables(i, 2));
end;
correlatedVariables = sortrows(correlatedVariables, [-3, 1, 2]);
correlatedVariables

% Eliminate the duplicate features


%% Detect and delete the outliers