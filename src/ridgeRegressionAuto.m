function [beta, trainingErr, testErr] = ridgeRegressionAuto(y, tX, K, lambdaValues, seed)
% Learn model parameters beta and best lambda using k-fold cross-validation
% Can plot the expected train and test error respective to lambda

% K: number of CV folds
% lambdaValues: values to try out as lambda parameters
% seed: used to reset the RNG and obtain comparable (identical) splits for
%       each k-fold CV (when comparing the lambdas)
    if(nargin < 3)
       K = 5;
    end;
    if(nargin < 4)
       lambdaValues = logspace(-2, 2, 100); 
    end;
    n = length(lambdaValues);
    
    trainingErr = zeros(n, 1);
    testErr = zeros(n, 1);
    bestErr = -1;
    
    % For each lambda value
    for i = 1:length(lambdaValues)
        lambda = lambdaValues(i);
        learn = @(y, tX) ridgeRegression(y, tX, lambda);
        predict = @(tX, beta) tX * beta;
        
        setSeed(seed);
        [trainingErr(i), testErr(i)] = kFoldCrossValidation(y, tX, K, learn, predict, @computeRmse);
        
        if(testErr(i) < bestErr || bestErr < 0)
            lambdaStar = lambda;
            bestErr = testErr(i);
            %fprintf('Error %f obtained with lambda = %f\n', bestErr, lambda);
        end;
    end;
    
    % We have now chosen the best lambda value, we can use all the provided
    % tX as train data to learn a model
    beta = ridgeRegression(y, tX, lambdaStar);
    
    % Plot evolution of train and test error with respect to lambda
    %{
    figure;
    semilogx(lambdaValues, trainingErr, '.-b');
    hold on;
    semilogx(lambdaValues, testErr, '.-r');
    xlabel('Lambda');
    ylabel('Training (blue) and test (red) error');
    %}
end