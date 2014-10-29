function y = normalized(x)
  means = ones(size(x,1), 1) * mean(x);
  deviations = ones(size(x,1), 1) * std(x);
  y = (x - means) ./ deviations;
end
