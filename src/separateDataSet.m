function [beta, tX1, y1, tX2, y2] = separateDataSet(y, tX, threshold)
% Assuming our output y can be explained by two distinct model,
% we try and learn a classifier on the input tX to distinguish between the
% two. We use the given threshold on the output value to create the two
% classes.
    c2 = (y > threshold);
    binary = false(size(y));
    binary(c2) = true();
    
    beta = logisticRegression(binary, tX);
    yPredicted = binaryPrediction(tX, beta);
    c1 = (yPredicted == 0);
    
    % Output the split dataset
    tX1 = tX(c1,:);
    y1 = y(c1);
    tX2 = tX(~c1,:);
    y2 = y(~c1);
    
    % Check the validity of the classifier visually
    %{
    figure;
    for i = 1:25
        subplot(5, 5, i);
        hold on;
        plot(tX1, y1, '.b');
        plot(tX2, y2, '.r');
    end;
    %}
end