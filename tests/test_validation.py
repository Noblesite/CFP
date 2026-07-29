from cfp.models import Severity
from cfp.validation import validate_workflow
from cfp.workflow import Workflow


def finding_codes(workflow: Workflow) -> set[str]:
    return {item.code for item in validate_workflow(workflow).findings}


def test_baseline_validates_without_errors(cfp02: Workflow) -> None:
    report = validate_workflow(cfp02)

    assert report.errors == []


def test_duplicate_node_ids_are_detected(cfp02: Workflow) -> None:
    workflow = cfp02.clone()
    workflow.nodes[1]["id"] = workflow.nodes[0]["id"]

    assert "duplicate_node_id" in finding_codes(workflow)


def test_duplicate_link_ids_are_detected(cfp02: Workflow) -> None:
    workflow = cfp02.clone()
    workflow.links[1][0] = workflow.links[0][0]

    assert "duplicate_link_id" in finding_codes(workflow)


def test_invalid_node_references_are_detected(cfp02: Workflow) -> None:
    workflow = cfp02.clone()
    workflow.links[0][1] = 999999

    assert "missing_origin_node" in finding_codes(workflow)


def test_missing_subgraph_definition_is_detected(cfp02: Workflow) -> None:
    workflow = cfp02.clone()
    missing_id = workflow.nodes[-1]["type"]
    workflow.data["definitions"]["subgraphs"] = [
        item for item in workflow.subgraphs if item["id"] != missing_id
    ]

    report = validate_workflow(workflow)
    assert any(
        item.code == "missing_subgraph_definition"
        and item.severity == Severity.ERROR
        for item in report.findings
    )

