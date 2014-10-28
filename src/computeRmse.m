function cost = computeRmse(y, yHat)
	cost = sqrt(2 * computeMse(y, yHat));
end

function err = computeMse(y, yHat)
	n = size(y, 1);
	e = y - yHat;
	err = (e' * e) / (2 * n);
end