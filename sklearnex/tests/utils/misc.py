from daal4py.sklearn._utils import sklearn_check_version

if sklearn_check_version("1.9"):
    from sklearn.utils._array_api import get_namespace_and_device

    __all__ = ["assert_same_namespace"]

    def assert_same_namespace(a, b):
        xp_a, _, device_a = get_namespace_and_device(a)
        xp_b, _, device_b = get_namespace_and_device(b)
        assert xp_a == xp_b
        assert device_a == device_b

else:
    __all__ = []
