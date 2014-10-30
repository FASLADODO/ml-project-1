function [y, y_test] = normalized(x, x_test)
  % need to keep means and deviations to use them for test data
  % normalization
  means = ones(size(x,1), 1) * mean(x);
  deviations = ones(size(x,1), 1) * std(x);
  y = (x - means) ./ deviations;
  
  % normalization of the test data with the previous means and deviations
  if (nargin > 1)
      n = size(x_test,1);
      y_test = (x_test - means(1:n,:)) ./ deviations(1:n,:);
  end
end
