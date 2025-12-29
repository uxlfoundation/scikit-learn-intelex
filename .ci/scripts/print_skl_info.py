import sys

import sklearn

if __name__ == "__main__":
    if (sys.version_info.major > 3) or (
        sys.version_info.major == 3 and sys.version_info.minor > 10
    ):
        sklearn.show_versions()
