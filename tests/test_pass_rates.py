import xml.etree.ElementTree as ET

import pytest

from scripts.pass_rates import outcome


@pytest.mark.parametrize(
    "xml, expected",
    [
        ('<testcase name="a"/>', "passed"),
        ('<testcase name="b"><failure message="assert False"/></testcase>', "failed"),
        ('<testcase name="c"><error message="fixture crashed"/></testcase>', "failed"),
        ('<testcase name="d"><skipped type="pytest.xfail" message="known"/></testcase>', "xfail"),
        ('<testcase name="e"><skipped type="pytest.skip" message="no reason"/></testcase>', "skipped"),
    ],
)
def test_outcome_reads_pytest_junit_results(xml, expected):
    # The xfail format was checked against real `pytest --junitxml` output.
    assert outcome(ET.fromstring(xml)) == expected
