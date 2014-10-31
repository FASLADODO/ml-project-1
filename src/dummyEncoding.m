function result = dummyEncoding(X, categoricalVariables)
% Perform dummy encoding of categorical features from X
% Categorical features are removed from the result and dummy-encoded
% features are added to the end.
%
% categoricalVariables (optionnal) A vector of indices of the categorical features

    if(nargin < 2)
        categoricalVariables = 1:size(X, 2);
    end;

    cat = false(size(X, 2), 1);
    cat(categoricalVariables) = true();
    
    result = X(:, ~cat);
    for i = categoricalVariables;
       result = [result dummyvar(X(:, i)+1)];
    end
end