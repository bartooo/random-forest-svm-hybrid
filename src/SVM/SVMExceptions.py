class SVMMinParamsException(Exception):
    """Raised when user provided keys for SVM Minimizer are not correct."""

    pass


class SVMPredMapperParamsException(Exception):
    """Raised when user provided keys for SVM Predictions Mapper are not correct."""

    pass


class SVMWrongDimExceptions(Exception):
    """Raised when provided dataset or model params are not compatible with model params."""

    pass


class SVMWrongTypeParamsExceptions(Exception):
    """Raised when provided parameters for model are wrong type."""

    pass


class SVMNotInitException(Exception):
    """Raised when user use predict() without model initialization/training."""

    pass


class SVMRFNoDataException(Exception):
    """Raised when user is using attributes drawing, but not specify X dataset in initialise method."""

    pass
