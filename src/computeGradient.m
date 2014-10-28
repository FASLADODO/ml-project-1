function [ g ] = computeGradient( y, tX, beta )
e = y - tX * beta;
g = (- 1/size(y, 1)) * tX' * e;
end

