"""
Authors: Bartosz Cywiński, Łukasz Staniszewski
"""
from sklearn.model_selection import train_test_split
from src.SVM.SVM import SVM
import numpy as np
import pytest
from src.SVM.SVMExceptions import (
    SVMMinParamsException,
    SVMPredMapperParamsException,
    SVMRFNoDataException,
    SVMWrongDimExceptions,
    SVMNotInitException,
    SVMWrongTypeParamsExceptions,
)
from sklearn.datasets import load_breast_cancer


def test_correct_tagets():
    targ_1_in = np.array([-3, 2, 0, -1, 1])
    targ_1_out = np.array([[-1], [1], [-1], [-1], [1]])
    assert SVM.correct_targets(targ_1_in).shape == targ_1_out.shape
    assert (SVM.correct_targets(targ_1_in) == targ_1_out).all()
    assert SVM.correct_targets(np.array([])) is not None


def test_svm_init_lambd():
    svm = SVM(lambd=3)
    assert svm.lambd is not None
    assert svm.lambd == 3
    assert svm.lambd != 0


def test_svm_init_min_params():
    svm = SVM(lambd=3)
    assert svm.beta is not None
    assert svm.max_steps is not None
    assert svm.min_epsilon is not None
    assert svm.beta == 0.01
    assert svm.max_steps == 10000
    assert svm.min_epsilon == 1e-20

    svm = SVM(lambd=3, minimizer_params={"beta": 1, "max_steps": 10})
    assert svm.beta is not None
    assert svm.max_steps is not None
    assert svm.min_epsilon is not None
    assert svm.beta == 1
    assert svm.max_steps == 10
    assert svm.min_epsilon == 1e-20

    svm = SVM(
        lambd=2,
        minimizer_params={
            "min_epsilon": 1e-10,
            "beta": 1,
            "max_steps": 10,
        },
    )
    assert svm.beta is not None
    assert svm.max_steps is not None
    assert svm.min_epsilon is not None
    assert svm.beta == 1
    assert svm.max_steps == 10
    assert svm.min_epsilon == 1e-10
    with pytest.raises(SVMMinParamsException):
        svm = SVM(lambd=2, minimizer_params={"wrong_key_name": 1e-10})
    with pytest.raises(SVMMinParamsException):
        svm = SVM(lambd=2, minimizer_params={"beta": "wrong_type_value"})


def test_svm_init_model():
    svm = SVM(lambd=3)
    svm.initialize_model(W=np.zeros(shape=(3, 1)), b=3)
    assert svm.b == 3
    assert (svm.W == np.array([[0], [0], [0]])).all()
    with pytest.raises(SVMWrongTypeParamsExceptions):
        svm.initialize_model(W=np.zeros(shape=(3, 1), dtype=np.int64), b=3)


def test_svm_init_mapper():
    svm = SVM(lambd=3)
    assert svm._mapper == {1: 1, -1: 0}
    svm = SVM(lambd=3, mapper_params={1: 1, -1: -1})
    assert svm._mapper != {1: 1, -1: 0}
    assert svm._mapper == {1: 1, -1: -1}
    with pytest.raises(SVMPredMapperParamsException):
        svm = SVM(lambd=3, mapper_params={"1": 1})
    with pytest.raises(SVMPredMapperParamsException):
        svm = SVM(lambd=3, mapper_params={1: "1"})
    with pytest.raises(SVMPredMapperParamsException):
        svm = SVM(lambd=3, mapper_params={3: 1})


def test_f():
    svm = SVM(lambd=3)
    svm.initialize_model(W=np.array([[1], [-1], [1]], dtype=np.float64), b=3)
    X = np.array([[1, 2, 3], [6, 5, 4], [7, 8, 7]])
    y_hat = np.array([[-1], [2], [3]])
    assert svm._f(X=X).shape == y_hat.shape
    assert (svm._f(X=X) == y_hat).all()
    with pytest.raises(SVMWrongDimExceptions):
        assert svm._f(X=np.array([[]])) is not None


def test_classify_y():
    svm = SVM(lambd=3)
    svm.initialize_model(W=np.array([[1], [-1], [1]], dtype=np.float64), b=3)
    X = np.array([[1, 2, 3], [6, 5, 4], [7, 8, 7]])
    y_hat = np.array([[-1], [1], [1]])
    assert svm._classify_y(X=X).shape == y_hat.shape
    assert (svm._classify_y(X=X) == y_hat).all()
    with pytest.raises(SVMWrongDimExceptions):
        assert svm._classify_y(X=np.array([[]])) is not None


def test_jacobian_f():
    svm = SVM(lambd=0.5)
    svm.initialize_model(W=np.array([[1], [-1], [1]], dtype=np.float64), b=3)
    X = np.array([[1, 2, 3], [6, 5, 4], [7, 8, 7]])
    Y = np.array([[1], [-1], [1]])
    delta_w_hat = np.array([[6], [2], [2]])
    delta_b_hat = 0.0
    delta_w, delta_b = svm._jacobian_f(X=X, y=Y)
    assert delta_w.shape == delta_w_hat.shape
    assert (delta_w_hat == delta_w).all()
    assert delta_b_hat == delta_b
    with pytest.raises(SVMWrongDimExceptions):
        assert svm._jacobian_f(X=np.array([[]]), y=np.array([[]])) is not None


def test_minimize():
    svm = SVM(lambd=0.5, minimizer_params={"beta": 0.2, "max_steps": 1})
    svm.initialize_model(W=np.array([[1], [-1], [1]], dtype=np.float64), b=3)
    X = np.array([[1, 2, 3], [6, 5, 4], [7, 8, 7]], dtype=np.float64)
    y = np.array([[1], [-1], [1]], dtype=np.float64)
    svm._gradient_descent_minimize(X=X, y=y)
    w_hat = np.array([[-0.2], [-1.4], [0.6]], dtype=np.float64)
    assert svm.b == 3
    assert svm.W.shape == w_hat.shape
    assert np.linalg.norm(svm.W - w_hat) < 1e-15


def test_fit():
    svm = SVM(lambd=0.5, minimizer_params={"beta": 0.2, "max_steps": 1})
    svm.initialize_model(W=np.array([[1], [-1], [1]], dtype=np.float64), b=3)
    X = np.array([[1, 2, 3], [6, 5, 4], [7, 8, 7]], dtype=np.float64)
    y = np.array([[1], [-1], [1]], dtype=np.float64)
    svm.fit(X=X, y=y, is_model_to_init=False)
    w_hat = np.array([[-0.2], [-1.4], [0.6]], dtype=np.float64)
    assert svm.b == 3
    assert svm.W.shape == w_hat.shape
    assert np.linalg.norm(svm.W - w_hat) < 1e-15


def test_fit_not_init():
    svm = SVM(lambd=0.5, minimizer_params={"beta": 0.2, "max_steps": 1})
    X = np.array([[1, 2, 3], [6, 5, 4], [7, 8, 7]], dtype=np.float64)
    y = np.array([[1], [-1], [1]], dtype=np.float64)
    with pytest.raises(SVMNotInitException):
        svm.fit(X=X, y=y, is_model_to_init=False)


def test_predict_default():
    svm = SVM(lambd=0.5, minimizer_params={"beta": 0.2, "max_steps": 1})
    svm.initialize_model(W=np.array([[1], [-1], [1]], dtype=np.float64), b=3)
    X = np.array([[1, 2, 3], [6, 5, 4], [7, 8, 7]], dtype=np.float64)
    y = np.array([[1], [0], [1]], dtype=np.float64)
    svm.fit(X=X, y=y, is_model_to_init=False)
    X_E = np.array([[-3, 4, 1], [4, 2, 12]], dtype=np.float64)
    Y_hat = np.array([0, 1], dtype=np.int32)
    assert svm.predict(X_E).shape == Y_hat.shape
    assert np.linalg.norm(svm.predict(X_E) - Y_hat) < 1e-13


def test_precict_not_initialized():
    svm = SVM(lambd=0.5)
    X_E = np.array([[-3, 4, 1], [4, 2, 12]], dtype=np.float64)
    with pytest.raises(SVMNotInitException):
        svm.predict(X=X_E)


def test_predict_usr_mapper():
    svm = SVM(
        lambd=0.5,
        minimizer_params={"beta": 0.2, "max_steps": 1},
        mapper_params={1: 1, -1: -1},
    )
    svm.initialize_model(W=np.array([[1], [-1], [1]], dtype=np.float64), b=3)
    X = np.array([[1, 2, 3], [6, 5, 4], [7, 8, 7]], dtype=np.float64)
    y = np.array([[1], [0], [1]], dtype=np.float64)
    svm.fit(X=X, y=y, is_model_to_init=False)
    X_E = np.array([[-3, 4, 1], [4, 2, 12]], dtype=np.float64)
    Y_hat = np.array([-1, 1], dtype=np.float64)
    assert svm.predict(X_E).shape == Y_hat.shape
    assert np.linalg.norm(svm.predict(X_E) - Y_hat) < 1e-15


def test_randomized_attributes():
    svm = SVM(lambd=3, rf_max_attributes=4)
    with pytest.raises(SVMWrongDimExceptions):
        svm.initialize_model(W=np.array([[1], [-1], [1]], dtype=np.float64), b=3)
    with pytest.raises(SVMRFNoDataException):
        svm.initialize_model(W=np.array([[1], [-1], [1], [3]], dtype=np.float64), b=3)
    X_E = np.array([[-3, 4, 1, 2]], dtype=np.float64)
    svm.initialize_model(
        W=np.array([[1], [-1], [1], [3]], dtype=np.float64), b=3, X=X_E
    )
    svm.predict(X_E)

    svm = SVM(lambd=3, rf_max_attributes=2)
    X = np.array([[1, 2, 3], [6, 5, 4], [7, 8, 7]], dtype=np.float64)
    y = np.array([[1], [0], [1]], dtype=np.float64)
    svm.fit(X=X, y=y)
    assert svm.W.shape[0] == 2
    X_E = np.array([[-3, 4, 1], [4, 2, 12]], dtype=np.float64)
    Y_hat = np.array([-1, 1], dtype=np.float64)
    assert svm.predict(X_E).shape == Y_hat.shape
    svm.predict(X_E)


def test_SVM_check_working_breast_cancer():
    data = load_breast_cancer()
    X = data.data
    y = data.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=0
    )
    svm = SVM(
        lambd=0.005,
        minimizer_params={"beta": 0.01, "min_epsilon": 1e-20},
    )
    svm.fit(X_train, y_train)
    preds = svm.predict(X_test)
    assert preds.shape[0] == y_test.shape[0]
    assert 0.5 < np.sum(preds == y_test) / preds.shape[0] <= 1.0
