import importlib
import subprocess
import sys
import warnings
from pathlib import Path

from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

ROOT = Path(__file__).resolve().parent.parent
STRICT = "import warnings; warnings.simplefilter('error')\n"
XML_DOC = '<?xml version="1.0" encoding="utf-8"?><package><item id="a"/></package>'


# ##################################################################
# test executing conftest installs the filter
# pytest resets warning filters per test, so execute the real conftest inside an isolated warnings scope
def test_executing_conftest_installs_filter() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        BeautifulSoup(XML_DOC, "html.parser")
        assert [w for w in caught if issubclass(w.category, XMLParsedAsHTMLWarning)]
        caught.clear()
        importlib.reload(importlib.import_module("src.conftest"))
        BeautifulSoup(XML_DOC, "html.parser")
        assert not [w for w in caught if issubclass(w.category, XMLParsedAsHTMLWarning)]


# ##################################################################
# test warning is raised without conftest
# a fresh interpreter that never imports the conftest turns the warning into an error
def test_warning_is_raised_without_conftest() -> None:
    code = f"{STRICT}from bs4 import BeautifulSoup; BeautifulSoup({XML_DOC!r}, 'html.parser')"
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=ROOT,
        check=False,
    )
    assert result.returncode != 0
    assert "XMLParsedAsHTMLWarning" in result.stderr


# ##################################################################
# test conftest suppresses in fresh interpreter
# importing the conftest makes the same strict-error interpreter parse cleanly
def test_conftest_suppresses_in_fresh_interpreter() -> None:
    code = (
        f"{STRICT}"
        "import src.conftest\n"
        "from bs4 import BeautifulSoup\n"
        f"BeautifulSoup({XML_DOC!r}, 'html.parser')\n"
        "print('clean')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=ROOT,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "clean"
