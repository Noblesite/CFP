import hashlib
from pathlib import Path

import pytest

from cfp.validation import validate_workflow
from cfp.workflow import Workflow

EXTERNAL_FIXTURES = {
    "High_Quality_GGUF.json": (
        "979137c6a5d8a0e510a638f5ce25879aae3e75a8000d87d6712559b9c5c01575"
    ),
    "Trellis2Multiviews_GGUF.json": (
        "69e2c7477ddd687862b9090e7acdb620a1e47e83880e9aa58c58fad03e1eded8"
    ),
}
EXTERNAL_FIXTURE_DIR = Path(__file__).parent / "fixtures" / "external"


@pytest.mark.parametrize(("filename", "expected_sha256"), EXTERNAL_FIXTURES.items())
def test_external_fixture_loads_validates_and_matches_checksum(
    filename: str,
    expected_sha256: str,
) -> None:
    path = EXTERNAL_FIXTURE_DIR / filename
    workflow = Workflow.load(path)

    assert hashlib.sha256(path.read_bytes()).hexdigest() == expected_sha256
    assert validate_workflow(workflow).errors == []


@pytest.mark.parametrize("filename", EXTERNAL_FIXTURES)
def test_inspection_does_not_add_missing_definitions(filename: str) -> None:
    workflow = Workflow.load(EXTERNAL_FIXTURE_DIR / filename)
    original = workflow.clone().data

    workflow.summary()

    assert "definitions" not in workflow.data
    assert workflow.data == original
