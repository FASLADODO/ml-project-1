function Xpoly = createPoly(X, degree)
% build matrix Phi for polynomial regression of a given degree
    N = size(X, 1); % Number of data examples
    D = size(X, 2); % Initial dimensionality of X
    
    Xpoly = [X zeros(N, (degree-1) * D)];
    for k = 2:degree
        for i = 1:D
            Xpoly(:, (k-1) * D + i) = X(:,i) .^ k;
        end
    end
end