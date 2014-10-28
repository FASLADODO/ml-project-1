function beta = logisticRegression(y, tX, alpha)
% Fit a linear model using Logistic Regression (using Newton's method)
% alpha: step size

	% algorithm parametesfor maximum iterations and convergence
	maxIters = 1000;
	convergence_th = 0; % convergence threshold

	% initialize
	beta = zeros(size(tX, 2), 1);

	L_last = 0;

	% iterate
	for k = 1:maxIters
		% compute loss, gradient and hessian
		[L, g, H] = logisticRegLoss(y, tX, beta);
		% update beta
		beta = beta - alpha * inv(H) * g;

		% check convergence
		if abs(L_last - L) < convergence_th
			break;
		end 

		L_last = L;
	end

end


function [L,g,H] = logisticRegLoss(y, tX, beta) 
	% pass tX*beta through sigmoid function
	sig = 1 ./ (1 + exp(-tX*beta));
	% compute S matrix
	s = diag(sig .* (1-sig));
	% compute hessian
	H = tX' * s * tX;
	% compute gradient 
	g = tX' * (sig - y);
	% compute negative log likelihood 
	L = - sum(y .* log(sig) + (1-y) .* log(1-sig));
end


