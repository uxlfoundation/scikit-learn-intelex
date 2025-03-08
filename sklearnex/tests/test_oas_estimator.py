import numpy as np
from sklearnex.covariance.oas_estimator import OASEstimator

def test_oas_estimator_fit():
    X = np.random.randn(100, 5)
    estimator = OASEstimator(shrinkage=0.1)
    estimator.fit(X)
    assert estimator.covariance_ is not None

def test_oas_estimator_score():
    X = np.random.randn(100, 5)
    estimator = OASEstimator(shrinkage=0.1)
    estimator.fit(X)
    score = estimator.score(X)
    assert isinstance(score, float)
