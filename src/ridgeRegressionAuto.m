function [betaStar, trainingErr, testErr] = ridgeRegressionAuto(y, tX, K, lambdaValues)
% Learn model parameters beta and best lambda using k-fold cross-validation
% Plot the error respective to lambda
%
% K: number of CV folds
% lambdaValues: values to try out as lambda parameters
    if(nargin < 3)
       K = 5;
    end;
    if(nargin < 4)
       lambdaValues = logspace(-2, 2, 100); 
    end;
    n = length(lambdaValues);
    
    % Split data in k folds (create indices only)
    N = size(y, 1);
    idx = randperm(N);
    Nk = floor(N / K);
    cvIndices = zeros(K, Nk);
    for k = 1:K
        cvIndices(k, :) = idx( (1 + (k-1)*Nk):(k * Nk) );
    end;
    
    trainingErr = zeros(n, 1);
    testErr = zeros(n, 1);
    bestErr = -1;
    
    % For each lambda value
    for i = 1:length(lambdaValues)
        lambda = lambdaValues(i);
        
        % For each fold, compute the train and test error
        subTrError = zeros(K, 1);
        subTeError = subTrError;
        for k = 1:K
            % Get k'th subgroup in test, others in train
            idxTe = cvIndices(k, :);
            idxTr = cvIndices([1:k-1 k+1:end], :);
            idxTr = reshape(idxTr, numel(idxTr), 1);
            yTe = y(idxTe);
            tXTe = tX(idxTe, :);
            yTr = y(idxTr);
            tXTr = tX(idxTr, :);

            % Learn model parameters
            beta = ridgeRegression(yTr, tXTr, lambda);

            % Compute training and test error for k'th train / test split
            subTrError(k) = computeRmse(yTr, tXTr * beta); 
            subTeError(k) = computeRmse(yTe, tXTe * beta); 
        end;

        % Estimate test and train errors for this lambda
        % are the average over the k folds
        trainingErr(i) = mean(subTrError);
        testErr(i) = mean(subTeError);

        if(testErr(i) < bestErr || bestErr < 0)
            betaStar = beta;
            bestErr = testErr(i);
            fprintf('Error %f obtained with lambda = %f\n', bestErr, lambda);
        end;
    end;
    
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