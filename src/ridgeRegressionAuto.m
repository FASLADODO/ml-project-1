function betaStar = ridgeRegressionAuto(y, tX, proportion, k, lambdaValues)
% Learn model parameters beta and best lambda using k-fold cross-validation
% Plot the error respective to lambda
%
% split: proportion of train vs data to use
% k: number of CV folds
% lambdaValues: values to try out as lambda parameters
    if(nargin < 3)
        proportion = 0.8;
    end;
    if(nargin < 4)
       k = 5; 
    end;
    if(nargin < 5)
       lambdaValues = logspace(-2, 2, 100); 
    end;
    n = length(lambdaValues);
    
    % Train / test split
    [tXTr, yTr, tXTe, yTe] = split(y, tX, proportion);
    
    trainingErr = zeros(n, 1);
    testErr = zeros(n, 1);
    bestErr = -1;
    for i = 1:length(lambdaValues)
        lambda = lambdaValues(i);
        
        % TODO: k-fold cross-validation
        beta = ridgeRegression(yTr, tXTr, lambda);
        
        trainingErr(i, :) = computeRmse(yTr, tXTr * beta);
        testErr(i, :) = computeRmse(yTe, tXTe * beta);

        if(testErr(i, :) < bestErr || bestErr < 0)
            betaStar = beta;
            bestErr = testErr(i, :);
            %fprintf('Error %f obtained with lambda = %f\n', bestErr, lambda);
        end;
    end;
    
    % Plot
    figure;
    semilogx(lambdaValues, trainingErr, '.-b');
    hold on;
    semilogx(lambdaValues, testErr, '.-r');
    xlabel('Lambda');
    ylabel('Training (blue) and test (red) error');
end