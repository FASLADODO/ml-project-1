function beta = penLogisticRegression(y, tX, alpha, lambda)
% Learn model parameters beta using Penalized Logistic Regression
% alpha: gradient descent step size
% lambda: regularization parameter

	% algorithm parametesfor maximum iterations and convergence
	maxIters = 1000;
	convergence_th = 0; % convergence threshold

	% initialize
	beta = zeros(size(tX, 2), 1);

	L_last = 0;

	% iterate
	for k = 1:maxIters
		% compute loss, gradient and hessian
		[L, g, H] = penLogisticRegLoss(y, tX, beta, lambda);
		% update beta
		beta = beta - alpha * inv(H) * g;

		% check convergence
		if abs(L_last - L) < convergence_th
			break;
		end 

		L_last = L;
	end

end



function [L,g,H] = penLogisticRegLoss(y, tX, beta, lambda) 
	% pass tX*beta through sigmoid function
	sig = 1 ./ (1 + exp(-tX*beta));
	% compute S matrix
	s = diag(sig .* (1-sig));
	% compute hessian
	H = tX' * s * tX + 2*lambda;
	% compute gradient 
	g = tX' * (sig - y) + 2*lambda*beta;
	% compute negative log likelihood 
	L = - sum(y .* log(sig) + (1-y) .* log(1-sig)) + lambda * beta * beta';
end


