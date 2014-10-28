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

% Removing the outliers
threshold = 10; % outliers are more than 10 standard deviation from the median
[X y] = removeOutliers(X,y,threshold);

%% Output Visualization
hist(y);

%% Input Visualization
figure;
side = 6;
for k = 1:size(X, 2)
    subplot(side, side, k);
    hist(X(:, k));
    title(['X', int2str(k)]);
end;


%% Plotting the features individually against Y
figure;
side = 6;
for k = 1:size(X, 2)
    subplot(side, side, k);
    plot(X(:, k), y, '.');
    title(['X', int2str(k), ' versus Y']);
end;

%% Plotting the features against each other

% figure;
% offset = 0;
% side = 10;
% for i = 1:side
%     for j  = 1:side
%         subplot(side, side, (i - 1) * side + j);
%         plot(X(:, i+offset), X(:, j+offset), '.');
%         title(['X', int2str(i+offset), ' versus X', int2str(j+offset)]);
%     end;
% end;


%% Compute the correlation between the features and spot the largest ones


correlatedVariables = computeFeaturesCorrelations(X);
% We obtain the highest correlation coefficients and the corresponding
% input variables indices
correlatedVariables

% We find some strong negative correlations as well as positive ones

% Eliminate the features which do not give information
%% Dummy variables

Xc = X(:,[30:end]);
Xnew = [];
for i = 1:size(Xc,2);
   Xdummy = dummyvar(Xc(:,i)+1);
   Xnew = [Xnew Xdummy];
end
X = [X(:,[1:29]) Xnew];

imagesc(X); colorbar;

% Now with dummy encoding we have binary variables from X30 to X46
% /!\ make sure you don't run that code several time or it will create
% other dummyvar from existing dummyvar