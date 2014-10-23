clear all;
addpath('../data');

load('regression.mat');
X = X_train;
y = y_train;

hist(y);
% there is one main gaussian + one other thing that we cannot understand :
% too much data to be outliers : a second gaussian maybe ?


%% Variables distributions look Gaussian. Variables from 36 are categorical
% -> should not be normalized

% Normalize data
X(:,1:35) = normalized(X(:,1:35));

% check Normalization
% Visualization
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
%%
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

% Handling categorical variables : Dummy encoding -> JHWT book + google it

