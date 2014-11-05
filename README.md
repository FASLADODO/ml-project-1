ml-project-1
============

EPFL's Pattern Classification and Machine Learning first course project

Team members
------------

- Jade Copet
- Merlin Nimier-David
- Krishna Sapkota

The project was designed by Prof. Emtiyaz & TAs. Some ML methods and helper functions were provided by the teaching team.

Project structure
-----------------

We provide the full project folder. The most relevant files are located in:

- `src`: the machine learning methods used to train model parameters
- `results`: the CSV outputs
- `report`: the report PDF and LaTeX source
- `exploration`: preliminary data analysis scripts developed to get ahold of the datasets
- `regression` and `classification`: dataset-specific scripts used to optimize expected test error (i.e. train the best model).

Project's TODO
--------------

### Data pre-processing

- [X] Try ridge regression on several seeds for different degrees and plot + check stability
- [X] Select a few degrees and do cross validation. (Selected degree 4 and output cool boxplots)
- [X] Try removing more features + compare stability with different methods and do cool boxplots. (Didn't seem to help)
- [X] Try increasing seeds number

### ML methods

- [X] `beta = leastSquaresGD(y,tX,alpha)`: Least squares using gradient descent (alpha is the step-size)
- [X] `beta = leastSquares(y,tX)`: Least squares using normal equations
- [X] `beta = ridgeRegression(y,tX, lambda)`: Ridge regression using normal equations (lambda is the regularization coefficient)
- [X] `beta = logisticRegression(y,tX,alpha)`: Logistic regression using gradient descent or Newton's method (alpha is the step size, in case of gradient descent)
- [X] `beta = penLogisticRegression(y,tX,alpha,lambda)`: Penalized logistic regression using gradient descent or Newton's method (alpha is the step size for gradient descent, lambda is the regularization parameter)
- [X] Implement cross-validation for ridge regression
- [X] Implement cross-validation for penalized logistic regression
- [X] Implement generic cross-validation to estimate test and train error for each method

### Regression dataset
- [X] Learn a classifier to separate the two data models
- [X] Minimize the train and test error
- [X] Check the stability of the results
- [X] Output predictions to CSV
- [X] Update report with our results

### Classification dataset
- [X] Verify / debug automatic penalized logistic regression
- [X] Minimize the train and test error
- [X] Check the stability of the results
- [X] Output predictions to CSV
- [X] Update report with our results

### Predictions
- [X] `predictions_regression.csv`: Each row contains prediction yhatn for a data example in the test set
- [X] `predictions_classification.csv`: Each row contains probability p(y=1|data) for a data example in the test set
- [X] `test_errors_regression.csv`: Report RMSE for methods "leastSquaresGD", leastSquares", "ridgeRegression"
- [X] `test_errors_classification.csv`: Report RMSE, 0-1 loss and log-loss for methods "logisticRegression", "penLogisticRegression"

### Report
- [X] Produce figures for the regression dataset
- [X] Report work done for the regression dataset and the corresponding results
- [X] Produce figures for the classification dataset
- [X] Double-check all figures for labels (on each axis and for the figure itself)
- [X] Clear conclusion and analysis of the results for each dataset
- [X] Include complete details about each algorithm (lambda values, number of folds, number of trials, etc)
- [X] What worked and what did not? Why do you think are the reasons behind that?
- [X] Why did you choose the method that you chose?
