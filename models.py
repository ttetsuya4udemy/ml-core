import numpy as np

class LinearRegressor:
    """
    A class that implements a simple linear regression model.
    The model calculates parameters using the normal equation and can be used for prediction.

    Attributes:
        theta (numpy.ndarray): The parameters (weights) of the regression model.
    """

    def __init__(self):
        """
        Initializes the LinearRegressor with default parameters.
        """
        self.theta = None

    def fit(self, X, y):
        """
        Fits the linear regression model to the provided data using the normal equation.

        Args:
            X (numpy.ndarray): A 2D array where each row represents a sample, and each column represents a feature.
            y (numpy.ndarray): A 1D array containing the target values corresponding to the samples in X.

        Example:
            >>> X = np.array([[1], [2], [3]])
            >>> y = np.array([2, 4, 6])
            >>> model = LinearRegressor()
            >>> model.fit(X, y)
        """
        X_b = np.c_[np.ones_like(X), X]
        self.theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

    def predict(self, X):
        """
        Predicts the target values for the given input features using the learned parameters.

        Args:
            X (numpy.ndarray): A 2D array where each row represents a sample, and each column represents a feature.

        Returns:
            numpy.ndarray: A 1D array containing the predicted values.

        Example:
            >>> X_test = np.array([[4], [5]])
            >>> predictions = model.predict(X_test)
        """
        X_b = np.c_[np.ones_like(X), X]
        return X_b.dot(self.theta)

    def get_params(self):
        """
        Returns the learned parameters (weights) of the model.

        Returns:
            numpy.ndarray: A 1D array containing the model parameters.

        Example:
            >>> params = model.get_params()
            >>> print(params)
        """
        return self.theta

