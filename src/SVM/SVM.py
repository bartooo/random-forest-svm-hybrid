import numpy as np


def correct_targets(targets: np.ndarray) -> np.ndarray:
    """Function correcting targets of dataset to be compatible with SVM.

    Args:
        targets (np.ndarray): array of targets with only -1 and 1 with shape (n,1)

    Returns:
        np.ndarray: array of targets from dataset
    """
    targets[np.where(targets > 0)] = 1
    targets[np.where(targets <= 0)] = -1
    return targets.reshape((targets.shape[0], 1))


class SVM:
    """Class represents SVM model.
    """

    def __init__(self, lambd: float, minimizer_params: dict = {}) -> None:
        """Class SVM constructor.

        Args:
            lambd (float): lambda penalty hiperparameter
            minimizer_params (dict, optional): parameters for gradient descent minimizer. 
                Defaults to {"beta": 0.01, "max_steps": 10000, "min_epsilon": 1e-20}.
        """
        self.params = None
        self.lambd = lambd
        self._set_minimizer_params(**minimizer_params)

    def _set_minimizer_params(self, **kwargs) -> None:
        """Function setting minimizer params.
        """
        self.beta = 0.01
        self.max_steps = 10000
        self.min_epsilon = 1e-20
        default_params = {
            "beta": self.beta,
            "max_steps": self.max_steps,
            "min_epsilon": self.min_epsilon,
        }
        for key, item in kwargs.items():
            assert key in default_params.keys()
            assert type(item) is int or type(item) is float
            default_params[key] = item

    def _gradient_descent_minimize(self, X: np.ndarray, y: np.ndarray) -> None:
        """Method performing optimization using gradient descent.

        Args:
            X (np.ndarray): matrix of samples with atributes
            y (np.ndarray): samples targets
        """
        step = 0
        while 1:
            gradient = self._jacobian_f(X=X, y=y)
            if np.linalg.norm(
                    gradient) < self.min_epsilon or step > self.max_steps:
                break
            self.params = self.params - self.beta * gradient
            step += 1

    def _f(self, X: np.ndarray) -> np.ndarray:
        """Method counts function value.

        Args:
            X (np.ndarray): matrix of samples with attributes

        Returns:
            np.ndarray: function values for samples
        """
        b = self.params[-1, :]
        W = self.params[:-1, :]
        return np.dot(X, W) - b

    def _classify_y(self, X: np.ndarray) -> np.ndarray:
        """Method correcting model out to {-1,1}.

        Args:
            X (np.ndarray): matrix of samples with attributes

        Returns:
            np.ndarray: targets for samples
        """
        return 2 * (self._f(X=X) > 0) - 1

    def _jacobian_f(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Method counts jacobian value.

        Args:
            X (np.ndarray): matrix of samples with attributes
            y (np.ndarray): targets for samples

        Returns:
            np.ndarray: jacobian values for samples
        """
        b = self.params[-1, :]
        W = self.params[:-1, :]

        # numpy array of partial derivatives
        partials = np.zeros_like(self.params, dtype=np.float64)

        # counting gradients for w1, w2, ..., wn
        distances = 1 - np.multiply(y, self._f(X))
        distances = distances.reshape((distances.shape[0], ))
        x_w_part = np.zeros_like(
            X)  # sum = 2 * λ * wi + (0 or iyx) over all samples
        x_w_part[np.where(distances > 0)] -= (y * X)[np.where(distances > 0)]
        partials[:-1, :] = 2 * self.lambd * W + np.sum(
            x_w_part, axis=0).reshape(partials[:-1, :].shape)

        # counting gradient for b
        x_b_part = np.zeros_like(y, dtype=np.float64)
        x_b_part[np.where(distances > 0)] += y[np.where(distances > 0)]
        partials[-1, :] = np.sum(x_b_part, axis=0)

        return partials

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Function performs fitting model to given dataset.

        Args:
            X (np.ndarray): matrix of samples with attributes
            y (np.ndarray): targets for samples
        """
        y = correct_targets(targets=y)
        self.params = np.zeros(shape=(X.shape[1] + 1, 1), dtype=np.float64)
        self._gradient_descent_minimize(X=X, y=y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Function performs prediction of model on given dataset.

        Args:
            X (np.ndarray): matrix of inputs with attributes

        Returns:
            np.ndarray: predictions for given inputs
        """
        return self._classify_y(X=X)
