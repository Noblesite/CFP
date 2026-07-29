from pathlib import Path

from cfp.workflow import Workflow


def test_round_trip_preserves_unknown_fields(
    cfp02: Workflow,
    tmp_path: Path,
) -> None:
    workflow = cfp02.clone()
    workflow.data["future_extension"] = {
        "opaque": [1, {"nested": True}],
    }
    workflow.nodes[0]["extension_owned"] = "preserve me"
    destination = tmp_path / "round-trip.json"

    workflow.save(destination)
    reloaded = Workflow.load(destination)

    assert reloaded.data == workflow.data

