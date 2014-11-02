function [RMSE, zero_one, logLoss] = computeErrorEstimate(XTe, yTe, beta)
	[yHat, pHat] = binaryPrediction(XTe, beta);

	% RMSE
	RMSE = sqrt((yTe - pHat)'*(yTe - pHat) / size(yTe, 1));

	% 0-1 loss
	zero_one = sum(yTe ~= yHat) / size(yTe, 1);
	

	% logLoss
	logLoss = - sum(yTe .* log(pHat) + (1-yTe) .* log(1-pHat)) / size(yTe, 1);
end