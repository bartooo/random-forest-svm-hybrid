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
        self.W = None
        self.b = None
        self.lambd = lambd
        self._set_minimizer_params(**minimizer_params)

    def _set_minimizer_params(self, **kwargs) -> None:
        """Function setting minimizer params.
        """
        self._set_beta(beta=0.01)
        self._set_max_steps(max_steps=10000)
        self._set_min_epsilon(epsilon=1e-20)
        default_params = {
            "beta": self._set_beta,
            "max_steps": self._set_max_steps,
            "min_epsilon": self._set_min_epsilon,
        }
        for key, item in kwargs.items():
            assert key in default_params.keys()
            assert type(item) is int or type(item) is float
            default_params[key](item)

    def initialize_model(self, W: np.ndarray, b: float):
        """Model parameters initializer.

        Args:
            W (np.ndarray): weights for attributes, should be np.float64 matrix of shape (model_dim, 1)
            b (float): bias for model parameters
        """
        assert W.shape[1] == 1
        assert W.dtype == np.float64
        self.W = W
        self.b = b

    def _set_beta(self, beta: float) -> None:
        """Beta setter.

        Args:
            beta (float): new optimizer's beta param
        """
        self.beta = beta

    def _set_max_steps(self, max_steps: int) -> None:
        """Max steps setter.

        Args:
            max_steps (int): new optimizer's max_steps param
        """
        self.max_steps = max_steps

    def _set_min_epsilon(self, epsilon: float) -> None:
        """Minimal epsilon setter.

        Args:
            epsilon (float): new optimizer's min_epsilon param
        """
        self.min_epsilon = epsilon

    def _gradient_descent_minimize(self, X: np.ndarray, y: np.ndarray) -> None:
        """Method performing optimization using gradient descent.

        Args:
            X (np.ndarray): matrix of samples with atributes
            y (np.ndarray): samples targets
        """
        step = 0
        while 1:
            step += 1
            gradient_W, gradient_b = self._jacobian_f(X=X, y=y)
            if np.linalg.norm(
                    gradient_W) < self.min_epsilon or step > self.max_steps:
                break
            self.W = self.W - self.beta * gradient_W
            self.b = self.b - self.beta * gradient_b

    def _f(self, X: np.ndarray) -> np.ndarray:
        """Method counts function value.

        Args:
            X (np.ndarray): matrix of samples with attributes

        Returns:
            np.ndarray: function values for samples
        """
        if len(X.shape) == 2:
            assert X.shape[1] == self.W.shape[0]
        return np.dot(X, self.W) - self.b

    def _classify_y(self, X: np.ndarray) -> np.ndarray:
        """Method correcting model out to {-1,1}.

        Args:
            X (np.ndarray): matrix of samples with attributes

        Returns:
            np.ndarray: targets for samples
        """
        return 2 * (self._f(X=X) > 0) - 1

    def _jacobian_f(self, X: np.ndarray,
                    y: np.ndarray) -> tuple[np.ndarray, float]:
        """Method counts jacobian value.

        Args:
            X (np.ndarray): matrix of samples with attributes
            y (np.ndarray): targets for samples

        Returns:
            tuple[np.ndarray, float]: tuple containing gradients w.r.t W and b
        """

        # numpy array of partial derivatives
        partials_W = np.zeros_like(self.W, dtype=np.float64)
        partial_b = 0.0

        # counting gradients for w1, w2, ..., wn
        distances = 1 - np.multiply(y, self._f(X))
        distances = distances.reshape((distances.shape[0], ))
        x_w_part = np.zeros_like(
            X)  # sum = 2 * λ * wi + (0 or iyx) over all samples
        x_w_part[np.where(distances > 0)] -= (y * X)[np.where(distances > 0)]
        partials_W = 2 * self.lambd * self.W + np.sum(
            x_w_part, axis=0).reshape(self.W.shape)

        # counting gradient for b
        x_b_part = np.zeros_like(y, dtype=np.float64)
        x_b_part[np.where(distances > 0)] += y[np.where(distances > 0)]
        partial_b = np.sum(x_b_part, axis=0)

        return partials_W, partial_b

    def fit(self, X: np.ndarray, y: np.ndarray, is_model_to_init: bool = True):
        """Function performs fitting model to given dataset.

        Args:
            X (np.ndarray): matrix of samples with attributes
            y (np.ndarray): targets for samples
            is_model_to_init (bool, optional): whether to initialize model parameters with zeros, defaults to True
        """
        y = correct_targets(targets=y)
        if is_model_to_init:
            self.initialize_model(W=np.zeros(shape=(X.shape[1], 1),
                                             dtype=np.float64),
                                  b=0.0)
        self._gradient_descent_minimize(X=X, y=y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Function performs prediction of model on given dataset.

        Args:
            X (np.ndarray): matrix of inputs with attributes

        Returns:
            np.ndarray: predictions for given inputs
        """
        return self._classify_y(X=X)
