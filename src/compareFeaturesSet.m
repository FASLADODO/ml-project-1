function res = compareFeaturesSet(y, X, Xl, seedsNb)
% Used to test results on Xl which is a cleaned version of X (without meaningless-suspected features). 
% Compare least squares results on two different features set on different seeds
    
    trErrLS_full = zeros(seedsNb,1);
    teErrLS_full = zeros(seedsNb,1);
    trErrLS_light = zeros(seedsNb,1);
    teErrLS_light = zeros(seedsNb,1);

    for i=1:seedsNb
        seed = i;

        % least Squares on X
        [X_train, y_train, X_test, y_test] = split(y, X, 0.8, seed);

        N = length(y_train);
        tX = [ones(N, 1) X_train];
        tX_test = [ones(length(y_test), 1) X_test];

        betaLS_full = leastSquares(y_train, tX);
        trErrLS_full(i,:) = computeRmse(y_train, tX * betaLS_full);
        teErrLS_full(i,:) = computeRmse(y_test, tX_test * betaLS_full);

        % least Squares on Xl
        [Xl_train, yl_train, Xl_test, yl_test] = split(y, Xl, 0.8, seed);

        Nl = length(yl_train);
        tXl = [ones(N, 1) Xl_train];
        tXl_test = [ones(length(yl_test), 1) Xl_test];

        betaLS_light = leastSquares(yl_train, tXl);
        trErrLS_light(i,:) = computeRmse(yl_train, tXl * betaLS_light);
        teErrLS_light(i,:) = computeRmse(yl_test, tXl_test * betaLS_light);
        
        % Printing results on each seed
        fprintf('Error with least squares: tr %f | te %f  VS tr %f | te %f\n', trErrLS_full(i,:), teErrLS_full(i,:), trErrLS_light(i,:), teErrLS_light(i,:));
    end

    % Plotting the comparison on train and test errors
    res = [trErrLS_full trErrLS_light teErrLS_full teErrLS_light];
    positions = [1 1.25 1.75 2];

    boxplot(res, 'notch','on', 'positions', positions);
    ylabel(['RMSE on ', int2str(seedsNb) ' seeds']);
    xtix = {'train X','train light X','test X','test light X'};
    xtixloc = [1 1.25 1.75 2];
    set(gca,'XTickMode','auto','XTickLabel',xtix,'XTick',xtixloc);

end