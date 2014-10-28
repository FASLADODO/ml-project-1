function cost = computeRmse(y, tX, beta)
	cost = sqrt(2 * computeMse(y, tX, beta));
end

function err = computeMse(y, tX, beta)
	n = size(y, 1);
	e = y - (tX * beta);
	err = (e' * e) / (2 * n);
end