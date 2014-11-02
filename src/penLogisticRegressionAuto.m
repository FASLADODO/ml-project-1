function [betaStar, trainingErr, testErr] = penLogisticRegressionAuto(y, tX, K, lambdaValues)
% Learn model parameters beta and best lambda using K-fold cross-validation
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
    % Step size
    alpha = 1e-3;
    
    trainingErr = zeros(n, 1);
    testErr = zeros(n, 1);
    bestErr = -1;

    % Split data in k fold (create indices only)
    setSeed(1);
    N = size(y, 1);
    idx = randperm(N);
    Nk = floor(N/K);
    cvIndices = zeros(K, Nk);
    for k = 1:K
        cvIndices(k, :) = idx( (1 + (k-1)*Nk):(k * Nk) );
    end
    
    % Tryout all lambda values
    % For each value, we train with Ridge Regression using cross validation
    for i = 1:n
        lambda = lambdaValues(i);
        
        % k-fold cross-validation
        mseTrSub = zeros(K, 1);
        mseTeSub = mseTrSub;
        for k = 1:K
            % Get k'th subgroup in test, others in train
            idxTe = cvIndices(k, :);
            idxTr = cvIndices([1:k-1 k+1:end], :);
            idxTr = reshape(idxTr, numel(idxTr), 1);
            yTe = y(idxTe);
            XTe = tX(idxTe, :);
            yTr = y(idxTr);
            XTr = tX(idxTr, :);

            % Train beta
            beta = penLogisticRegression(yTr, XTr, alpha, lambda);

            % Compute training and test error for k'th train / test split
            mseTrSub(k) = computeLogisticRegressionMse(yTr, XTr, beta); 
            mseTeSub(k) = computeLogisticRegressionMse(yTe, XTe, beta); 
        end
        
        % Training and test error for this lambda value is the average over
        % all k cross-validation trials
        trainingErr(i) = mean(mseTrSub);
        testErr(i) = mean(mseTeSub);

        size(trainingErr);
        
        % Best beta is the one for which the average CV test error is the least
        if(testErr(i) < bestErr || bestErr < 0)
            betaStar = beta;
            bestErr = testErr(i, :);
        end;
        
        % Status
        %fprintf('Error for lambda = %f: %f | %f\n', lambda, trainingErr(i), testErr(i));
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




