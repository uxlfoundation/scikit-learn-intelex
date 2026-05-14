from abc import ABC, abstractmethod

from ..utils._array_api import get_namespace


# Adds common methods to create oneDAL objects after a fallback
class _BaseLinearModel(ABC):
    @abstractmethod
    def _initialize_onedal_estimator(self, override_fit_intercept: bool = False) -> None:
        raise NotImplementedError()

    def _get_fit_intercept(self, override_fit_intercept: bool = False) -> bool:
        if not override_fit_intercept:
            return self.fit_intercept
        else:
            if isinstance(self.intercept_, float):
                return self.intercept_ != 0.0
            else:
                xp, _ = get_namespace(self.coef_)
                return bool(xp.all(self.intercept_ != 0))

    def _initialize_onedal_estimator_from_coefs(self) -> None:
        xp, _ = get_namespace(self.coef_)
        self._initialize_onedal_estimator(override_fit_intercept=True)
        self._onedal_estimator._create_model(self.coef_, self.intercept_, xp)
