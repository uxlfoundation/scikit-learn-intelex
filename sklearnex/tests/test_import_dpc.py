import os
import pathlib


def test_import_onedal_py_dpc():
    import onedal

    print(os.listdir(pathlib.Path(onedal.__file__).parent.resolve()))
    print("MKLROOT:", os.environ.get("MKLROOT", None))
    import onedal._onedal_py_dpc
