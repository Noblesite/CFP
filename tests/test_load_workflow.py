import hashlib
from pathlib import Path

from cfp.workflow import Workflow

CFP02_SHA256 = "854a49ca4a3bff5d22de7794803fe0098baf9fdefc76df26d779bf02278a23d7"


def test_cfp02_loads(cfp02_path: Path) -> None:
    workflow = Workflow.load(cfp02_path)

    assert workflow.summary() == {
        "workflow_id": "baf7dd25-6e83-47a9-9e09-1faec90306e7",
        "version": 0.4,
        "revision": 0,
        "nodes": 11,
        "links": 11,
        "groups": 7,
        "subgraphs": 3,
        "last_node_id": 227,
        "last_link_id": 11,
    }


def test_cfp02_fixture_is_byte_for_byte_unchanged(cfp02_path: Path) -> None:
    digest = hashlib.sha256(cfp02_path.read_bytes()).hexdigest()

    assert digest == CFP02_SHA256


def test_load_rejects_non_object_root(tmp_path: Path) -> None:
    path = tmp_path / "invalid.json"
    path.write_text("[]", encoding="utf-8")

    try:
        Workflow.load(path)
    except ValueError as error:
        assert "root must be an object" in str(error)
    else:
        raise AssertionError("Expected a non-object workflow root to be rejected")
