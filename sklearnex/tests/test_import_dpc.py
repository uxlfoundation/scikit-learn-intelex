import os
import pathlib


def test_import_onedal_py_dpc():
    import onedal

    print(os.listdir(pathlib.Path(onedal.__file__).parent.resolve()))
    print("MKLROOT:", os.environ.get("MKLROOT", None))
    print("PATH:", os.environ.get("PATH", None))
    print(
        "MKL contents:",
        os.listdir(
            "D://a//scikit-learn-intelex//scikit-learn-intelex//oneapi//mkl//latest//bin"
        ),
    )
    import onedal._onedal_py_dpc
