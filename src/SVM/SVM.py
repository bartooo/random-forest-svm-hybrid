"""
Authors: Bartosz Cywiński, Łukasz Staniszewski
"""
import numpy as np
from src.SVM.SVMExceptions import (
    SVMMinParamsException,
    SVMPredMapperParamsException,
    SVMWrongDimExceptions,
    SVMNotInitException,
    SVMWrongTypeParamsExceptions,
    SVMRFNoDataException,
)
from src.SVM.constants import (
    MINIMIZER_PARAMS_DEFAULT,
    PRED_MAPPER_PARAMS_DEFAULT,
)
import copy
from typing import Tuple


class SVM:
    """Class represents SVM model."""

    def __init__(
        self,
        lambd: float,
        minimizer_params: dict = {},
        mapper_params: dict = {},
        rf_max_attributes: int = None,
    ) -> None:
        """Class SVM constructor.

        Args:
            lambd (float): lambda penalty hiperparameter
            minimizer_params (dict, optional): parameters for gradient descent minimizer.
                Defaults to {"beta": 0.01, "max_steps": 10000, "min_epsilon": 1e-20}.
            mapper_params (dict, optional): parameters predictions mapper.
                Defaults to {1: 1, -1: 0}.
            rf_max_attributes (int, optional): max number of attributes used as model in random forest.
                Defaults to None, then all columns are used.
        """
        self.W = None
        self.b = None
        self.lambd = lambd
        self._set_minimizer_params(**minimizer_params)
        self._set_mapper_params(mapper_params)
        self.rf_max_attributes = rf_max_attributes
        self.used_columns_idx = None

    def _set_minimizer_params(self, **kwargs) -> None:
        """Function setting minimizer params."""
        self._set_beta(beta=MINIMIZER_PARAMS_DEFAULT["beta"])
        self._set_max_steps(max_steps=MINIMIZER_PARAMS_DEFAULT["max_steps"])
        self._set_min_epsilon(epsilon=MINIMIZER_PARAMS_DEFAULT["min_epsilon"])
        default_params = {
            "beta": self._set_beta,
            "max_steps": self._set_max_steps,
            "min_epsilon": self._set_min_epsilon,
        }
        for key, item in kwargs.items():
            if key not in default_params.keys():
                raise SVMMinParamsException(
                    "Possible keys in dict are only 'beta', 'max_steps'"
                    " and 'min_epsilon'!"
                )
            if type(item) is not int and type(item) is not float:
                raise SVMMinParamsException(
                    "Possible values for minimizer params dict items"
                    " are int or float!"
                )
            default_params[key](item)

    def _set_mapper_params(self, usr_mapped_params: dict) -> None:
        """Function setting mapper params.

        Args:
            usr_mapped_params (dict): params for mapper provided by user
        """
        mapper_params = copy.copy(PRED_MAPPER_PARAMS_DEFAULT)
        for key, item in usr_mapped_params.items():
            if key not in mapper_params.keys():
                raise SVMPredMapperParamsException(
                    "Possible keys in dict are only 1 and -1!"
                )
            if type(key) is not int or type(item) is not int:
                raise SVMPredMapperParamsException(
                    "Keys and items in dict should be int!"
                )
            mapper_params[key] = item
        self._mapper = mapper_params

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
            if np.linalg.norm(gradient_W) < self.min_epsilon or step > self.max_steps:
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
        if len(X.shape) != 2 or X.shape[1] != self.W.shape[0]:
            raise SVMWrongDimExceptions(
                "Provided X is not in shape (n, 1) or X columns not"
                " matching model parameters!"
            )
        return np.dot(X, self.W) - self.b

    def _classify_y(self, X: np.ndarray) -> np.ndarray:
        """Method correcting model out to {-1,1}.

        Args:
            X (np.ndarray): matrix of samples with attributes

        Returns:
            np.ndarray: targets for samples
        """
        return 2 * (self._f(X=X) > 0) - 1

    def _jacobian_f(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, float]:
        """Method counts jacobian value.

        Args:
            X (np.ndarray): matrix of samples with attributes
            y (np.ndarray): targets for samples

        Returns:
            tuple[np.ndarray, float]: tuple containing gradients w.r.t. W and b
        """

        # numpy array of partial derivatives
        partials_W = np.zeros_like(self.W, dtype=np.float64)
        partial_b = 0.0

        # counting gradients for w1, w2, ..., wn
        distances = 1 - np.multiply(y, self._f(X))
        distances = distances.reshape((distances.shape[0],))
        x_w_part = np.zeros_like(X)  # sum = 2 * λ * wi + (0 or iyx) over all samples
        x_w_part[np.where(distances > 0)] -= (y * X)[np.where(distances > 0)]
        partials_W = 2 * self.lambd * self.W + np.sum(x_w_part, axis=0).reshape(
            self.W.shape
        )

        # counting gradient for b
        x_b_part = np.zeros_like(y, dtype=np.float64)
        x_b_part[np.where(distances > 0)] += y[np.where(distances > 0)]
        partial_b = np.sum(x_b_part, axis=0)

        return partials_W, partial_b

    def _set_used_attributes(self, X: np.ndarray) -> None:
        """Method setting used by algorithm attributes.
        Used for random forest hybrid.

        Args:
            X (np.ndarray): dataset, from which columns attributes will be drawn
        """
        if self.rf_max_attributes:
            self.used_columns_idx = np.random.choice(
                range(X.shape[1]), self.rf_max_attributes, replace=False
            )
        else:
            self.used_columns_idx = np.linspace(
                0, X.shape[1] - 1, num=X.shape[1], dtype=np.int32
            )

    def _check_model_initialization(
        self, W: np.ndarray, b: float, X: np.ndarray = None
    ) -> None:
        if len(W.shape) != 2 or W.shape[1] != 1:
            raise SVMWrongDimExceptions(
                "Provided parameters for model should be in shape" " (n, 1)!"
            )
        if W.dtype != np.float64:
            raise SVMWrongTypeParamsExceptions(
                "Provided parameters for model should be np.float64!"
            )
        if self.rf_max_attributes and self.rf_max_attributes != W.shape[0]:
            raise SVMWrongDimExceptions(
                "Provided W param dont match number of attributes"
                " passed for random forest in constructor:"
                f" {W.shape[0]} != {self.rf_max_attributes}"
            )
        if self.used_columns_idx is None and self.rf_max_attributes and X is None:
            raise SVMRFNoDataException(
                "Dataset X not provided although rf_max_attributes"
                " provided in SVM constructor."
            )

    def initialize_model(self, W: np.ndarray, b: float, X: np.ndarray = None) -> None:
        """Model parameters initializer.

        Args:
            W (np.ndarray): weights for attributes,
                should be np.float64 matrix of shape (model_dim, 1)
            b (float): bias for model parameters
            X (np.ndarray, optional): dataset, from which columns attributes will be drawn.
                Defaults to None.
        """
        self._check_model_initialization(W=W, b=b, X=X)
        if self.used_columns_idx is None and self.rf_max_attributes and X is not None:
            self._set_used_attributes(X=X)
        self.W = W
        self.b = b
        self.W = W
        self.b = b

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        is_model_to_init: bool = True,
    ):
        """Function performs fitting model to given dataset.

        Args:
            X (np.ndarray): matrix of samples with attributes
            y (np.ndarray): targets for samples
            is_model_to_init (bool, optional): whether to initialize model parameters with zeros,
                set to False if model was initialized beforehand. Defaults to True.
        """
        if (self.W is None or self.b is None) and not is_model_to_init:
            raise SVMNotInitException(
                "You have to init model before fit or set" " is_model_to_init to True."
            )
        y_copy = self.correct_targets(targets=copy.copy(y))
        self._set_used_attributes(X=X)
        X_sliced = copy.copy(X[:, self.used_columns_idx])
        if is_model_to_init:
            self.initialize_model(
                W=np.zeros(shape=(X_sliced.shape[1], 1), dtype=np.float64),
                b=0.0,
            )
        self._gradient_descent_minimize(X=X_sliced, y=y_copy)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Function performs prediction of model on given dataset.

        Args:
            X (np.ndarray): matrix of inputs with attributes

        Returns:
            np.ndarray: predictions for given inputs
        """
        if self.W is None or self.b is None:
            raise SVMNotInitException("SVM is neither initialized nor fitted.")
        X_sliced = copy.copy(X)
        if len(X_sliced.shape) == 1:
            X_sliced = np.expand_dims(X_sliced, axis=0)
        if self.rf_max_attributes:
            X_sliced = X_sliced[:, self.used_columns_idx]
        return np.vectorize(self._mapper.__getitem__)(
            self._classify_y(X=X_sliced)
        ).squeeze()

    @staticmethod
    def correct_targets(targets: np.ndarray) -> np.ndarray:
        """Function correcting targets of dataset to be compatible with SVM.

        Args:
            targets (np.ndarray): array of targets with only -1 and 1 with shape (n,1)

        Returns:
            np.ndarray: array of targets from dataset
        """
        targets = np.where(targets > 0, 1, -1)
        return targets.reshape((targets.shape[0], 1))
