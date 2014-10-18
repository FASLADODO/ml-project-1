function y = normalized(x)
  means = ones(length(x), 1) * mean(x);
  deviations = ones(length(x), 1) * std(x);
  y = (x - means) ./ deviations;
end
