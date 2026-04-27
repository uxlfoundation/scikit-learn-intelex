import array_api_strict
import numpy as np

from sklearnex import config_context
from sklearnex.linear_model import LinearRegression


def test_non_writeable_arrays():
    rng = np.random.default_rng(seed=123)
    X = rng.random(size=(20, 4))
    y = rng.random(size=X.shape[0])
    X.flags.writeable = False
    y.flags.writeable = False
    Xs = array_api_strict.asarray(X)
    ys = array_api_strict.asarray(y)
    with config_context(array_api_dispatch=True):
        model = LinearRegression().fit(Xs, ys)
        _ = model.predict(Xs)
        _ = model.score(Xs, ys)
