import json
from pathlib import Path

from cfp.builders import append_part_isolation_stage, synchronize_kontext_prompts
from cfp.models import NodeOutputRef, PartIsolationStage
from cfp.validation import validate_workflow
from cfp.workflow import Workflow

SWAN_PROMPT = (
    "Using this elegant style, create a portrait of a swan wearing a pearl "
    "tiara and lace collar, maintaining the same refined quality and soft "
    "color tones."
)


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
    assert SWAN_PROMPT not in json.dumps(workflow.data)


def test_prompt_repair_preserves_outer_prompts_and_unrelated_quarantine(
    cfp02: Workflow,
) -> None:
    workflow = cfp02.clone()
    outer_prompts = {
        node["id"]: node["widgets_values"][0]
        for node in workflow.nodes
        if node.get("type") in {item["id"] for item in workflow.subgraphs}
    }

    report = synchronize_kontext_prompts(workflow)

    assert len(report.updated) == 3
    assert SWAN_PROMPT not in json.dumps(workflow.data)
    definitions = {item["id"]: item for item in workflow.subgraphs}
    for node_id, prompt in outer_prompts.items():
        node = workflow.node(node_id)
        assert node["widgets_values"][0] == prompt
        quarantine = node["properties"].get("proxyWidgetErrorQuarantine", [])
        assert all(
            entry.get("originalEntry", [None])[-1] != "text"
            for entry in quarantine
        )
        assert any(
            entry.get("originalEntry", [None])[-1] == "control_after_generate"
            for entry in quarantine
        )

        definition = definitions[node["type"]]
        clip = next(
            item for item in definition["nodes"] if item.get("type") == "CLIPTextEncode"
        )
        assert clip["widgets_values"][0] == prompt


def test_prompt_repair_does_not_promote_known_template_prompt(
    cfp02: Workflow,
) -> None:
    workflow = cfp02.clone()
    workflow.node(189)["widgets_values"][0] = SWAN_PROMPT

    report = synchronize_kontext_prompts(workflow)

    assert report.details["unresolved_template_prompt_nodes"] == [189]


def test_prompt_repair_can_restore_from_known_good_node_prompts(
    cfp02: Workflow,
) -> None:
    workflow = cfp02.clone()
    expected = workflow.node(189)["widgets_values"][0]
    workflow.node(189)["widgets_values"][0] = SWAN_PROMPT

    report = synchronize_kontext_prompts(
        workflow,
        prompt_overrides={189: expected},
    )

    assert report.details["unresolved_template_prompt_nodes"] == []
    assert workflow.node(189)["widgets_values"][0] == expected


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
