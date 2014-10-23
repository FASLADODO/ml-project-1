% Solve exercise : 

% start with some visualisation to learn more on data
% then apply simplest thing to the data
% then more complicated things and see if it?s going better


% load training data

hist(y,30) -> see outliers

% remove outliers : 
idx =  find(y(:)<8);
y = y(idx);
X = X(idx,:);

% to know which ones that I removed : (y(:)>8)

% Visualize 
% plot hist : which ones look gaussian, categorical?  => hist(X(:,i),30)
% I should rescale my data but not the categorical ones => normalizisation => plot them again to be sure I have done the good things

% Correlation between input and output :
% plot : seems that there is correlation between all the variables and the output except 3rd variable
% compute correlation coefficient instead of graphical result if we have too many variables

%Recode the categorical variables

%Split data and apply simplest methods : mean

err = loss(y, hat)
err = mean((y(:)-yhat(:)).^2)

rmseTr = loss(yTr, mean(yTr)); 
rmseTe=loss(yTe, mean(yTr))
%/!\ don?t train on test data !

% Dummy variables
Xc = X(:,[9:end]);
Xnew = [];
for i = 1:size(Xc,2);
   Xdummy = dummyvar(Xc(:,i)+1);
   Xnew = [Xnew Xdummy];
end
X = [X(:,[1:8]) Xnew];
   
imagesc(X); colorbar;
break;

% cross-correlations 
nr = 3; nc = 3;
for i = 1:size(X1,2)
    for j = 1:size(X1,2);
        subplot(nr,nc,h);
        plot(X1(:,i),X1(:,j),'o');
        title(sprintf('%d',i));
    end
    pause
end

% variable highly correlated to each others -> ill-conditionning. Here we
% see that 3rd variable is not correlated to other ones => then we can
% conclude that it is useless and not use it. 

% if we see that a bunch of variables are correlated to each others and
% correlated to output, we could ask only one of them in order to have a
% simpler model

% apply leastSquares
% Rank deficient : columns very dependent -> check how bad it is :
% rank(XTr) VS size(XTr) ex : 6 VS 10 -> we're loosing 4 degrees here

% if the output is not correlated to the output, you shouldn't ignore it :
% it might be that the combination of 2 variables makes the correlation. We
% don't want to just take all variables one by one
tXTr = 
beta = leasSquares
yhatTr = tXTe*beta;
yhatTe = tXTe*beta;

rmseTe = loss(yTr, yhatTr)
rmseTr = loss(yTe, yhatTe)

% there is an improvement by including input variables using leastsquares.
% Also X is ill-conditioned, must use Ridge Regression

% if we are lazy and don't want to do cross validation but to check if
% we're doing right, setSeet on the split and see if the results are stable
% on the different seeds

% check if we are not penalizing beta0 in our Ridge Regression test

% Ridge regression : range for CV seems to be from 10^(-2,2). Optimal value
% of lambda is 1 to 20.

% try some features transformation with basis functions see if it improves
% the model (check the error). ex: sqrt(X) seems to help -> write about how
% it has been improved, write on lambda, plot lambda and everything