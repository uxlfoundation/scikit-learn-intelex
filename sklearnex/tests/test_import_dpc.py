import os


def test_import_onedal_py_dpc():
    import onedal
    import onedal._onedal_py_host

    print(os.listdir(pathlib.Path(onedal._onedal_py_host.__file__).parent.resolve()))
    print("MKLROOT:", os.environ.get("MKLROOT", None))
    import onedal._onedal_py_dpc
