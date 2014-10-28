function cost = computeRmse(y, tX, beta)
	cost = sqrt(2 * computeMse(y, tX, beta));
end

