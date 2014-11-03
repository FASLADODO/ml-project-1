function err = computeErrorEstimate(XTe, yTe, beta)
	[yHat, pHat] = binaryPrediction(XTe, beta);
	
	% to avoid computing log0 in logLoss 
	pHat(pHat == 0) = 0.01;
	pHat(pHat == 1) = 0.99;

	% RMSE
	RMSE = sqrt((yTe - pHat)'*(yTe - pHat) / size(yTe, 1));

	% 0-1 loss
	zero_one = sum(yTe ~= yHat) / size(yTe, 1);
	
	% logLoss
	logLoss = - sum(yTe .* log(pHat) + (1-yTe) .* log(1-pHat)) / size(yTe, 1);

	err = [RMSE zero_one logLoss];
end