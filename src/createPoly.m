function Xpoly = createPoly(X,degree)
% build matrix Phi for polynomial regression of a given degree
    Xpoly = [];
    for i = 1:size(X,2)
        for k = 1:degree
            pol(:,k) = X(:,i).^k;
        end
        Xpoly = [Xpoly pol];
    end
end