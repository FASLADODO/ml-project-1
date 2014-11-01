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
       lambdaValues = logspace(0, 1, 100); 
    end;

    n = length(lambdaValues);
    alpha = 1e-3; % Step size
    
    trainingErr = zeros(n, 1);
    testErr = zeros(n, 1);
    bestErr = -1;

    % split data in k fold (create indices only)
    setSeed(1);
    N = size(y,1);
    idx = randperm(N);
    Nk = floor(N/K);
    for k = 1:K
        idxCV(k,:) = idx(1+(k-1)*Nk:k*Nk);
    end
    
    % tryout all lambda values
    for i = 1:n
        lambda = lambdaValues(i);
        
        % k-fold cross-validation
        for k = 1:K
            % get k'th subgroup in test, others in train
            idxTe = idxCV(k,:);
            idxTr = idxCV([1:k-1 k+1:end],:);
            idxTr = idxTr(:);
            yTe = y(idxTe);
            XTe = tX(idxTe,:);
            yTr = y(idxTr);
            XTr = tX(idxTr,:);

            % train beta on training data 
            beta = penLogisticRegression(yTr, XTr, alpha, lambda);

            % compute training and test MSE
            mseTrSub(k) = computeLogisticRegressionMse(yTr, XTr, beta); 
            mseTeSub(k) = computeLogisticRegressionMse(yTe, XTe, beta); 

        end

        trainingErr(i, :) = mean(mseTrSub);
        testErr(i, :) = mean(mseTeSub);

        if(testErr(i, :) < bestErr || bestErr < 0) % best beta is the one for which the test error is the least
            betaStar = beta;
            bestErr = testErr(i, :);
        end;
    end;
    
    % Plot evolution of train and test error with respect to lambda
    figure;
    semilogx(lambdaValues, trainingErr, '.-b');
    hold on;
    semilogx(lambdaValues, testErr, '.-r');
    xlabel('Lambda');
    ylabel('Training (blue) and test (red) error');

end



function setSeed(seed)
% set seed
    global RNDN_STATE  RND_STATE
    RNDN_STATE = randn('state');
    randn('state',seed);
    RND_STATE = rand('state');
    %rand('state',seed);
    rand('twister',seed);
end

