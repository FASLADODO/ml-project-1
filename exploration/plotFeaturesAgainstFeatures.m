function plotFeaturesAgainstFeatures(X, selector, features)
% Plot all features against each other one by one using a series of graphs.
% selector (optionnal) A logical vector selecting the data examples
%                      to separate the dataset in two colors
% features (optionnal) A vector of the indices of the features to use
    if(nargin < 2)
        selector = true(size(X, 1), 1);
    end;
    if(nargin < 3)
        d = size(X, 2);
        features = 1:d;
    end;
    
    disp('Plotting input variables against each other.');
    disp('Press any key to move on to the next variable. Close the figure stop.');

    side = ceil(sqrt(d));
    for reference = features
        disp(['Plotting X', int2str(reference), ' against all other features']);

        for i = 1:d
            subplot(side, side, i);
            plot(X(selector, reference), X(selector, i), '.b');
            hold on;
            plot(X(~selector, reference), X(~selector,i), '+r');
            title(['X', int2str(reference), ' versus X', int2str(i)]);
        end;

        waitforbuttonpress;
        clf;
    end;

end