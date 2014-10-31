function res = compareFeaturesSet(y, X, Xl, nSeeds)
% COMPAREFEATURESSET Compare two feature sets with Least Squares
% We compare the error given by these two datasets after learning a linear
% classifier using least squares. We run the comparison on nSeeds trials,
% with a different random train / test split at each trial.
%
% y Output (N x 1)
% X One feature set (N x D)
% Xl Another feature set to be compared (N x D'), e.g. with some features
% removed or transformations applied.
% nSeeds Number of trials to run
    
    trErr1 = zeros(nSeeds,1);
    teErr1 = zeros(nSeeds,1);
    trErr2 = zeros(nSeeds,1);
    teErr2 = zeros(nSeeds,1);

    for i = 1:nSeeds
        seed = i;

        % ----- X (trial i)
        [X_train, y_train, X_test, y_test] = split(y, X, 0.8, seed);

        N = length(y_train);
        tX = [ones(N, 1) X_train];
        tX_test = [ones(length(y_test), 1) X_test];

        betaLS_full = leastSquares(y_train, tX);
        trErr1(i,:) = computeRmse(y_train, tX * betaLS_full);
        teErr1(i,:) = computeRmse(y_test, tX_test * betaLS_full);

        % ----- Xl (trial i)
        [Xl_train, yl_train, Xl_test, yl_test] = split(y, Xl, 0.8, seed);

        Nl = length(yl_train);
        tXl = [ones(Nl, 1) Xl_train];
        tXl_test = [ones(size(Xl_test, 1), 1) Xl_test];

        betaLS_light = leastSquares(yl_train, tXl);
        trErr2(i,:) = computeRmse(yl_train, tXl * betaLS_light);
        teErr2(i,:) = computeRmse(yl_test, tXl_test * betaLS_light);
        
        % Results of trial i
        %fprintf('Error with least squares: tr %f | te %f  VS tr %f | te %f\n', trErr1(i,:), teErr1(i,:), trErr2(i,:), teErr2(i,:));
    end
    
    % Print average error and variance of the results
    fprintf('\n----- Results over %d trials\n', nSeeds);
    fprintf('Average error:\n');
    fprintf('- X (train): %f, X (test): %f\n', mean(trErr1), mean(teErr1));
    fprintf('- Xl (train): %f, Xl (test): %f\n', mean(trErr2), mean(teErr2));
    fprintf('Variance:\n');
    fprintf('- X (train): %f, X (test): %f\n', var(trErr1), var(teErr1));
    fprintf('- Xl (train): %f, Xl (test): %f\n', var(trErr2), var(teErr2));
    
    % Plotting train and test errors over all trials
    res = [trErr1 trErr2 teErr1 teErr2];
    positions = [1 1.25 1.75 2];

    boxplot(res, 'notch','on', 'positions', positions);
    ylabel(['RMSE on ', int2str(nSeeds) ' seeds']);
    xtix = {'train X','train light X','test X','test light X'};
    xtixloc = [1 1.25 1.75 2];
    set(gca,'XTickMode','auto','XTickLabel',xtix,'XTick',xtixloc);
end