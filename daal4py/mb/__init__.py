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

from sklearn.linear_model import LogisticRegression, SGDClassifier

from .logistic_regression_builders import LogisticDAALModel
from .tree_based_builders import GBTDAALBaseModel, GBTDAALModel

__all__ = ["LogisticDAALModel", "GBTDAALModel", "convert_model"]


def convert_model(model) -> "GBTDAALModel | LogisticDAALModel":
    """
    Convert GBT or LogReg models to Daal4Py

    This function can be used to convert machine learning models / estimators
    created through other libraries to daal4py classes which offer accelerated
    prediction methods.

    It supports gradient-boosted decision tree ensembles (GBT) from the libraries
    ``xgboost``, ``lightgbm``, ``catboost``, and ``treelite``; and logistic regression
    (binary and multinomial) models from scikit-learn.

    See the documentation of the classes :obj:`daal4py.mb.GBTDAALModel` and
    :obj:`daal4py.mb.LogisticDAALModel` for more details.

    As an alternative to this function, models of a specific type (GBT or LogReg)
    can also be instantiated by calling those classes directly - for example,
    logistic regression models can be instantiated directly from fitted coefficients
    and intercepts, thereby allowing to work with models from libraries beyond
    scikit-learn.

    Parameters
    ----------
    model : fitted model object
        A fitted model object (either GBT or LogReg) from the supported libraries.

    Returns
    -------
    obj : GBTDAALModel or LogisticDAALModel
        A daal4py model object of the corresponding class for the model type, which
        offers faster prediction methods.
    """
    if isinstance(model, LogisticRegression):
        if model.classes_.shape[0] > 2:
            if (model.multi_class == "ovr") or (
                model.multi_class == "auto" and model.solver == "liblinear"
            ):
                raise TypeError(
                    "Supplied 'model' object is a linear classifier, but not multinomial logistic"
                    " (hint: pass multi_class='multinomial' to 'LogisticRegression')."
                )
        elif (model.classes_.shape[0] == 2) and (model.multi_class == "multinomial"):
            raise TypeError(
                "Supplied 'model' object is not a logistic regressor "
                "(hint: pass multi_class='auto' to 'LogisticRegression')."
            )
        return LogisticDAALModel(model.coef_, model.intercept_)
    if isinstance(model, SGDClassifier):
        if model.classes_.shape[0] > 2:
            raise TypeError(
                "Supplied 'model' object is a linear classifier, but not multinomial logistic"
                " (note: scikit-learn does not offer stochastic multinomial logistic models)."
            )
        if model.loss != "log_loss":
            raise TypeError(
                "Supplied 'model' object is not a logistic regressor "
                "(hint: pass loss='log_loss' to 'SGDClassifier')."
            )
        return LogisticDAALModel(model.coef_, model.intercept_)

    return GBTDAALModel(model)
