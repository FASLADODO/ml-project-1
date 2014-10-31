addpath(genpath('./data'), genpath('../data'));
addpath(genpath('./src'), genpath('../src'));

%% Data pre-processing
clear;
load('classification.mat');

X = X_train;
y = y_train;

% We have N = 1500, D = 32
size(X);
size(y);

% categorical variables : X1, X15, X30 -> move it at the end of the X
% matrix : now X30, X31, X32 are categorical variables
X = [X(:,2:14) X(:,16:29) X(:,31:end) X(:,1) X(:,15) X(:,30)];

X(:,1:29) = normalized(X(:,1:29));

%% Output Visualization
hist(y);

% class selector
t = y > 0;
% We have 1052 example belonging to class y==1 and 448 examples belonging
% to class y==-1. Same as checking :
size(y(t))
size(y(~t))

%% Removing the outliers
threshold = 10; % outliers are more than 10 standard deviation from the median
[X, y] = removeOutliers(X, y, threshold);
t = y > 0;

%% Input Visualization
figure;
side = 6;
for k = 1:size(X, 2)
    subplot(side, side, k);
    hist(X(:, k));
    title(['X', int2str(k)]);
end;

% This is not very helpful.

%% Plotting the features individually against Y

figure;
side = 6;
for k = 1:size(X, 2)
    subplot(side, side, k);
    plot(X(t, k), y(t), '.r'); hold on;
    plot(X(~t, k), y(~t), '.b');
    title(['X', int2str(k), ' versus Y']);
end;

%% Plotting the features against each other
% added : class added in color (test for spotting interesting clues)
figure;
plotFeaturesAgainstFeatures(X, t);

%% Compute the correlation between the features and spot the largest ones

selector = @(x) abs(x) > 0.4;
correlatedVariables = findCorrelations(selector, X);
% We obtain the highest correlation coefficients and the corresponding
% input variables indices
correlatedVariables

% We find some strong negative correlations as well as positive ones

% Eliminate the features which do not give information
%% Dummy variables encoding for categorical input variables

categoricalVariables = 30:size(X, 2);
X = dummyEncoding(X, categoricalVariables);
imagesc(X); colorbar;

% Now with dummy encoding we have binary variables from X30 to X46
% /!\ make sure you don't run that code several time or it will create
% other dummyvar from existing dummyvar

%% k-nearest neighbors with matlab functions
% mdl = fitcknn(X,y,'NumNeighbors',18)
% rloss = resubLoss(mdl)
% cvmdl = crossval(mdl,'kfold',5)
% kloss = kfoldLoss(cvmdl)