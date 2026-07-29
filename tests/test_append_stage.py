from pathlib import Path

from cfp.builders import append_part_isolation_stage
from cfp.models import NodeOutputRef, PartIsolationStage
from cfp.validation import validate_workflow
from cfp.workflow import Workflow


def faceplate_stage() -> PartIsolationStage:
    return PartIsolationStage(
        part_id="faceplate",
        display_name="Helmet Faceplate",
        source_image_1=NodeOutputRef(191),
        source_image_2=NodeOutputRef(227),
        output_prefix="CFP-03/faceplate_candidate",
        seed=915231124983052,
    )


def test_append_part_isolation_stage_is_well_formed(cfp02: Workflow) -> None:
    workflow = cfp02.clone()
    original_subgraph_ids = {item["id"] for item in workflow.subgraphs}

    report = append_part_isolation_stage(workflow, faceplate_stage())

    assert len(workflow.nodes) == 14
    assert len(workflow.links) == 15
    assert len(workflow.groups) == 9
    assert len(workflow.subgraphs) == 4
    assert len({node["id"] for node in workflow.nodes}) == len(workflow.nodes)
    assert len({link[0] for link in workflow.links}) == len(workflow.links)
    assert len({item["id"] for item in workflow.subgraphs}) == len(workflow.subgraphs)
    assert workflow.subgraphs[-1]["id"] not in original_subgraph_ids
    assert workflow.nodes[-3]["type"] == workflow.subgraphs[-1]["id"]
    assert workflow.nodes[-3]["inputs"][0]["link"] == 12
    assert workflow.nodes[-3]["inputs"][1]["link"] == 13
    assert workflow.nodes[-2]["inputs"][0]["link"] == 14
    assert workflow.nodes[-1]["inputs"][0]["link"] == 15
    assert workflow.nodes[-3]["outputs"][0]["links"] == [14, 15]
    assert 12 in workflow.node(191)["outputs"][0]["links"]
    assert 13 in workflow.node(227)["outputs"][0]["links"]
    assert workflow.data["last_node_id"] == 230
    assert workflow.data["last_link_id"] == 15
    assert report.details["node_ids"] == [228, 229, 230]
    assert validate_workflow(workflow).errors == []


def test_append_does_not_mutate_source_fixture(
    cfp02_path: Path,
) -> None:
    before = cfp02_path.read_bytes()
    workflow = Workflow.load(cfp02_path)

    append_part_isolation_stage(workflow, faceplate_stage())

    assert cfp02_path.read_bytes() == before


def test_generated_file_reloads_and_validates(
    cfp02: Workflow,
    tmp_path: Path,
) -> None:
    workflow = cfp02.clone()
    append_part_isolation_stage(workflow, faceplate_stage())
    destination = tmp_path / "generated.json"

    workflow.save(destination)
    reloaded = Workflow.load(destination)

    assert validate_workflow(reloaded).errors == []

