function result = dummyEncoding(X, categoricalVariables)
% Perform dummy encoding of categorical features from X
% Categorical features are removed from X and dummy-encoded
% features are added at the end, forming result
%
% categoricalVariables (optional) A vector of indices of the categorical features

    if(nargin < 2)
        categoricalVariables = 1:size(X, 2);
    end;

    cat = false(size(X, 2), 1);
    cat(categoricalVariables) = true();
    
    % Warning: dummyvar outputs one extra feature which could be deduced
    % from the other (k-1). We remove it to avoid obtaining a rank
    % deficient matrix.
    
    result = X(:, ~cat);
    for i = categoricalVariables;
        dummy = dummyvar(X(:, i) + 1);
        result = [result dummy(:, 1:(size(dummy, 2)-1))];
    end
end