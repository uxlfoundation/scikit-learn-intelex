# Copyright contributors to the oneDAL project
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import re
import sys

import numpy as np

from .. import (
    classifier_prediction_result,
    logistic_regression_model_builder,
    logistic_regression_prediction,
)

_docstring_X = """Parameters
----------
X : array-like(n_samples, n_features)
    The features / covariates for each row. Can be passed as either a NumPy array
    or as a sparse CSR array/matrix from SciPy. For faster results, use the same
    dtype as what this object was built for."""
if (sys.version_info.major == 3) and (sys.version_info.minor <= 12):
    _docstring_X = re.sub("^", " " * 8, _docstring_X, flags=re.MULTILINE).strip()


class LogisticDAALModel:
    """
    Logistic Regression Predictor

    Creates a logistic regression or multionomial logistic regression model object
    which can calculate fast predictions of different types (classes, probabilities,
    logarithms of probabilities), from fitted coefficients and intercepts obtained
    elsewhere (such as from :obj:`sklearn.linear_model.LogisticRegression`), making
    the predictions either in double (``np.float64``) or single (``np.float32``)
    precision.

    See Also
    --------
    :obj:`sklearn.linear_model.LogisticRegression`, :obj:`sklearn.linear_model.SGDClassifier`,
    :obj:`daal4py.classifier_prediction_result`.

    Parameters
    ----------
    coefs : array(n_classes, n_features) or array(n_features,)
        The fitted model coefficients. Note that only dense arrays are supported.
        In the case of binary classification, can be passed as a 1D array or as a
        2D array having a single row.
    intercepts: array(n_classes) or float
        The fitted intercepts. In the case of binary classification, must be passed
        as either a scalar, or as a 1D array with a single entry.
    dtype : np.float32 or np.float64
        The dtype to use for the object.

    Attributes
    ----------
    n_classes_ : int
        Number of classes in the model.
    n_features_in_ : int
        Number of features in the model.
    dtype_ : np.dtype
        The dtype of the model
    coef_ : array(n_classes, n_features)
        The model coefficients
    intercept_ : array(n_classes)
        The model intercepts
    """

    def __init__(self, coefs, intercepts, dtype=np.float64):
        assert dtype in [np.float32, np.float64]
        coefs = np.require(coefs, requirements=["ENSUREARRAY"])
        if len(coefs.shape) == 1:
            coefs = coefs.reshape((1, -1))
        self.n_features_in_ = coefs.shape[1]
        self.n_classes_ = max(2, coefs.shape[0])
        intercepts = np.require(intercepts, requirements=["ENSUREARRAY"]).reshape(-1)
        if self.n_classes_ == 2:
            assert len(intercepts) == 1
        else:
            assert intercepts.shape[0] == coefs.shape[0]
        self._fptype = "float" if dtype == np.float32 else "double"
        self.dtype_ = dtype
        if coefs.dtype != self.dtype_:
            coefs = coefs.astype(self.dtype_)
        if intercepts.dtype != self.dtype_:
            intercepts = intercepts.astype(self.dtype_)
        builder = logistic_regression_model_builder(
            n_classes=self.n_classes_, n_features=coefs.shape[1]
        )
        builder.set_beta(coefs, intercepts)
        self._model = builder.model
        self._alg_pred_class = logistic_regression_prediction(
            nClasses=self.n_classes_,
            fptype=self._fptype,
            resultsToEvaluate="computeClassLabels",
        )
        self._alg_pred_prob = logistic_regression_prediction(
            nClasses=self.n_classes_,
            fptype=self._fptype,
            resultsToEvaluate="computeClassProbabilities",
        )
        self._alg_pred_logprob = logistic_regression_prediction(
            nClasses=self.n_classes_,
            fptype=self._fptype,
            resultsToEvaluate="computeClassLogProbabilities",
        )

    @property
    def coef_(self):
        return self._model.Beta[:, 1:]

    @property
    def intercept_(self):
        return self._model.Beta[:, 0]

    def predict(self, X) -> np.ndarray:
        """
        Predict most probable class

        %docstring_X%

        Returns
        -------
        classes : array(n_samples,)
            The most probable class, as integer indexes
        """
        return (
            self._alg_pred_class.compute(X, self._model)
            .prediction.reshape(-1)
            .astype(int)
        )

    predict.__doc__ = predict.__doc__.replace(r"%docstring_X%", _docstring_X)

    def predict_proba(self, X) -> np.ndarray:
        """
        Predict probabilities of belonging to each class

        %docstring_X%

        Returns
        -------
        proba : array(n_samples, n_classes)
            The predicted probabilities for each class.
        """
        return self._alg_pred_prob.compute(X, self._model).probabilities

    predict_proba.__doc__ = predict_proba.__doc__.replace(r"%docstring_X%", _docstring_X)

    def predict_log_proba(self, X) -> np.ndarray:
        """
        Predict log-probabilities of belonging to each class

        %docstring_X%

        Returns
        -------
        log_proba : array(n_samples, n_classes)
            The logarithms of the predicted probabilities for each class.
        """
        return self._alg_pred_logprob.compute(X, self._model).logProbabilities

    predict_log_proba.__doc__ = predict_log_proba.__doc__.replace(
        r"%docstring_X%", _docstring_X
    )

    def predict_multiple(
        self, X, classes: bool = True, proba: bool = True, log_proba: bool = True
    ) -> classifier_prediction_result:
        """
        Make multiple prediction types at once

        A method that can output the results from ``predict``, ``predict_proba``, and ``predict_log_proba``
        all together in the same call more efficiently than computing them independently.

        %docstring_X%
        classes : bool
            Whether to output class predictions (what is obtained from :meth:`predict`).
        proba : bool
            Whether to output per-class probability predictions (what is obtained from
            :meth:`predict_proba`).
        log_proba : bool
            Whether to output per-class logarithms of probabilities (what is obtained
            from :meth:`predict_log_proba`).

        Returns
        -------
        predictions : classifier_prediction_result
            An object of class :obj:`daal4py.classifier_prediction_result` with the requested
            prediction types for the same ``X`` data.
        """
        pred_request = "|".join(
            (["computeClassLabels"] if classes else [])
            + (["computeClassProbabilities"] if proba else [])
            + (["computeClassLogProbabilities"] if log_proba else [])
        )
        if not len(pred_request):
            raise ValueError(
                "Must request at least one of 'classes', 'proba', 'log_proba'."
            )
        return logistic_regression_prediction(
            nClasses=self.n_classes_,
            fptype=self._fptype,
            resultsToEvaluate=pred_request,
        ).compute(X, self._model)

    predict_multiple.__doc__ = predict_multiple.__doc__.replace(
        r"%docstring_X%", _docstring_X
    )
