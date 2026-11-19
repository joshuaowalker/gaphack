"""The package version: gaphack/version.py is the source in code, and matches pyproject.toml."""
import re
from pathlib import Path

import gaphack
from gaphack import version


def test_version_sources_agree():
    pyproject = (Path(__file__).parent.parent / "pyproject.toml").read_text()
    declared = re.search(r'^version = "([^"]+)"', pyproject, re.M).group(1)
    assert version.__version__ == declared
    assert gaphack.__version__ == declared
