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
    
    trErrLS_full = zeros(nSeeds,1);
    teErrLS_full = zeros(nSeeds,1);
    trErrLS_light = zeros(nSeeds,1);
    teErrLS_light = zeros(nSeeds,1);

    for i = 1:nSeeds
        seed = i;

        % ----- X (trial i)
        [X_train, y_train, X_test, y_test] = split(y, X, 0.8, seed);

        N = length(y_train);
        tX = [ones(N, 1) X_train];
        tX_test = [ones(length(y_test), 1) X_test];

        betaLS_full = leastSquares(y_train, tX);
        trErrLS_full(i,:) = computeRmse(y_train, tX * betaLS_full);
        teErrLS_full(i,:) = computeRmse(y_test, tX_test * betaLS_full);

        % ----- Xl (trial i)
        [Xl_train, yl_train, Xl_test, yl_test] = split(y, Xl, 0.8, seed);

        Nl = length(yl_train);
        tXl = [ones(N, 1) Xl_train];
        tXl_test = [ones(Nl, 1) Xl_test];

        betaLS_light = leastSquares(yl_train, tXl);
        trErrLS_light(i,:) = computeRmse(yl_train, tXl * betaLS_light);
        teErrLS_light(i,:) = computeRmse(yl_test, tXl_test * betaLS_light);
        
        % Results of trial i
        fprintf('Error with least squares: tr %f | te %f  VS tr %f | te %f\n', trErrLS_full(i,:), teErrLS_full(i,:), trErrLS_light(i,:), teErrLS_light(i,:));
    end

    % Plotting train and test errors over all trials
    res = [trErrLS_full trErrLS_light teErrLS_full teErrLS_light];
    positions = [1 1.25 1.75 2];

    boxplot(res, 'notch','on', 'positions', positions);
    ylabel(['RMSE on ', int2str(nSeeds) ' seeds']);
    xtix = {'train X','train light X','test X','test light X'};
    xtixloc = [1 1.25 1.75 2];
    set(gca,'XTickMode','auto','XTickLabel',xtix,'XTick',xtixloc);
end