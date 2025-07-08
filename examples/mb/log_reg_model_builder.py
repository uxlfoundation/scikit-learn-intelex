# ==============================================================================
# Copyright 2020 Intel Corporation
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
# ===============================================================================

import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

import daal4py as d4p
from daal4py.sklearn._utils import daal_check_version


def main():
    X, y = load_iris(return_X_y=True)
    n_classes = 3

    # set parameters and train
    clf = LogisticRegression(fit_intercept=True, max_iter=1000, random_state=0).fit(X, y)

    # convert model to d4p
    d4p_model = d4p.mb.convert_model(clf)

    # compute class predictions
    predict_result_d4p = d4p_model.predict(X)
    predict_result_sklearn = clf.predict(X)
    np.testing.assert_equal(
        predict_result_d4p,
        predict_result_sklearn,
    )

    # compute probability predictions
    predict_proba_result_d4p = d4p_model.predict_proba(X)
    predict_proba_result_sklearn = clf.predict_proba(X)
    np.testing.assert_allclose(
        predict_proba_result_d4p,
        predict_proba_result_sklearn,
    )

    # compute logarithms of probabilities
    predict_log_proba_result_d4p = d4p_model.predict_log_proba(X)
    predict_log_proba_result_sklearn = clf.predict_log_proba(X)
    np.testing.assert_allclose(
        predict_log_proba_result_d4p,
        predict_log_proba_result_sklearn,
    )

    # compute multiple prediction types at once
    pred_all = d4p_model.predict_multiple(
        X,
        classes=True,
        proba=True,
        log_proba=True,
    )
    np.testing.assert_almost_equal(
        pred_all.prediction.reshape(-1),
        predict_result_sklearn,
    )
    np.testing.assert_almost_equal(
        pred_all.probabilities,
        predict_proba_result_sklearn,
    )
    np.testing.assert_almost_equal(
        pred_all.logProbabilities,
        predict_log_proba_result_sklearn,
    )

    return (
        d4p_model,
        predict_result_d4p,
        predict_proba_result_d4p,
        predict_log_proba_result_d4p,
    )


if __name__ == "__main__":
    if daal_check_version(((2021, "P", 1))):
        (
            d4p_model,
            predict_result_d4p,
            predict_proba_result_d4p,
            predict_log_proba_result_d4p,
        ) = main()
        print("\nLogistic Regression coefficients:\n", d4p_model.coef_)
        print("\nLogistic Regression intercepts:\n", d4p_model.intercept_)
        print(
            "\nLogistic regression prediction results (first 10 rows):\n",
            predict_result_d4p[0:10],
        )
        print(
            "\nLogistic regression prediction probabilities (first 10 rows):\n",
            predict_proba_result_d4p[0:10],
        )
        print(
            "\nLogistic regression prediction log probabilities (first 10 rows):\n",
            predict_log_proba_result_d4p[0:10],
        )
        print("All looks good!")
