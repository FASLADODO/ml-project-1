ml-project-1
============

EPFL's Pattern Classification and Machine Learning first course project

## TO DO Project

### Data pre-processing

- [X] Try ridge regression on several seeds for different degrees and plot + check stability
- [X] Select a few degrees and do cross validation. (Selected degree 4 and output cool boxplots)
- [X] Try removing more features + compare stability with different methods and do cool boxplots. (Didn't seem to help)
- [ ] Try increasing seeds number

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

## Methods to be implemented

- `beta = leastSquaresGD(y,tX,alpha)`: Least squares using gradient descent (alpha is the step-size)
- `beta = leastSquares(y,tX)`: Least squares using normal equations
- `beta = ridgeRegression(y,tX, lambda)`: Ridge regression using normal equations (lambda is the regularization coefficient)
- `beta = logisticRegression(y,tX,alpha)`: Logistic regression using gradient descent or Newton's method (alpha is the step size, in case of gradient descent)
- `beta = penLogisticRegression(y,tX,alpha,lambda)`: Penalized logistic regression using gradient descent or Newton's method (alpha is the step size for gradient descent, lambda is the regularization parameter)
