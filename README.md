ml-project-1
============

EPFL's Pattern Classification and Machine Learning first course project

## TO DO Project

### Data pre-processing

- [X] Try ridge regression on several seeds for different degrees and plot + check stability
- [X] Select a few degrees and do cross validation. (Selected degree 4 and output cool boxplots)
- [X] Try removing more features + compare stability with different methods and do cool boxplots. (Didn't seem to help)
- [X] Try increasing seeds number

### ML methods
- [ ] Implement cross-validation for ridge regression
- [X] Implement cross-validation for penalized logistic regression

### Regression dataset
- [X] Learn a classifier to separate the two data models
- [X] Minimize the train and test error
- [X] Check the stability of the results
- [X] Output predictions to CSV
- [X] Update report with our results

### Classification dataset
- [ ] Verify / debug automatic penalized logistic regression
- [ ] Minimize the train and test error
- [ ] Check the stability of the results
- [ ] Output predictions to CSV
- [ ] Update report with our results

### Predictions
- [ ] `predictions_regression.csv` : Each row contains prediction yhatn for a data example in the test set
- [ ] `predictions_classification.csv` : Each row contains probability p(y=1|data) for a data example in the test set
- [ ] `test_errors_regression.csv` : Report RMSE for methods "leastSquaresGD", leastSquares", "ridgeRegression"
- [ ] `test_errors_classification.csv` : Report RMSE, 0-1 loss and log-loss for methods "logisticRegression", "penLogisticRegression"

### Report
- [X] Produce figures for the regression dataset
- [X] Report work done for the regression dataset and the corresponding results
- [ ] Produce figures for the classification dataset
- [ ] What worked and what did not? Why do you think are the reasons behind that?
- [ ] Why did you choose the method that you choose?
- [ ] Include complete details about each algorithm (lambda values, number of folds, number of trials, etc)
- [ ] Clear conclusion and analysis of the results for each dataset
- [ ] Double-check all figures for labels (on each axis and for the figure itself)

## Methods implemented

- `beta = leastSquaresGD(y,tX,alpha)`: Least squares using gradient descent (alpha is the step-size)
- `beta = leastSquares(y,tX)`: Least squares using normal equations
- `beta = ridgeRegression(y,tX, lambda)`: Ridge regression using normal equations (lambda is the regularization coefficient)
- `beta = logisticRegression(y,tX,alpha)`: Logistic regression using gradient descent or Newton's method (alpha is the step size, in case of gradient descent)
- `beta = penLogisticRegression(y,tX,alpha,lambda)`: Penalized logistic regression using gradient descent or Newton's method (alpha is the step size for gradient descent, lambda is the regularization parameter)
