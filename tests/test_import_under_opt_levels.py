import os
import subprocess
import sys

import pytest


@pytest.mark.parametrize("level", [1, 2, 3])
@pytest.mark.parametrize("module", ["daal4py", "onedal", "sklearnex"])
def test_import_under_different_opt_levels(level, module):
    subprocess.run(
        [sys.executable, "-c", f"import {module}"],
        check=True,
        env=os.environ | {"PYTHONOPTIMIZE": str(level)},
    )
