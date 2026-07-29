from __future__ import annotations

import copy
import uuid
from typing import Any

from cfp.models import ChangeReport, NodeOutputRef
from cfp.workflow import Workflow


def _template_generation_node(workflow: Workflow) -> tuple[dict[str, Any], dict[str, Any]]:
    definitions = {item.get("id"): item for item in workflow.subgraphs}
    for node in reversed(workflow.nodes):
        definition = definitions.get(node.get("type"))
        input_names = [item.get("name") for item in node.get("inputs", [])]
        if definition is not None and input_names[:2] == ["image1", "image2"]:
            return node, definition
    raise ValueError("No two-reference Kontext subgraph node was found")


def _template_node(workflow: Workflow, node_type: str) -> dict[str, Any]:
    for node in workflow.nodes:
        if node.get("type") == node_type:
            return node
    raise ValueError(f"No {node_type} node was found to use as a template")


def _append_source_link(
    workflow: Workflow,
    source: NodeOutputRef,
    target: dict[str, Any],
    target_slot: int,
    link_id: int,
) -> None:
    source_node = workflow.node(source.node_id)
    outputs = source_node.get("outputs", [])
    if source.output_slot < 0 or source.output_slot >= len(outputs):
        raise ValueError(
            f"Node {source.node_id} has no output slot {source.output_slot}"
        )
    links = outputs[source.output_slot].get("links")
    if links is None:
        links = []
        outputs[source.output_slot]["links"] = links
    links.append(link_id)
    target["inputs"][target_slot]["link"] = link_id
    workflow.links.append(
        [
            link_id,
            source.node_id,
            source.output_slot,
            target["id"],
            target_slot,
            outputs[source.output_slot].get("type", "IMAGE"),
        ]
    )


def _append_node_link(
    workflow: Workflow,
    origin: dict[str, Any],
    origin_slot: int,
    target: dict[str, Any],
    target_slot: int,
    link_id: int,
) -> None:
    output = origin["outputs"][origin_slot]
    if output.get("links") is None:
        output["links"] = []
    output["links"].append(link_id)
    target["inputs"][target_slot]["link"] = link_id
    workflow.links.append(
        [
            link_id,
            origin["id"],
            origin_slot,
            target["id"],
            target_slot,
            output.get("type", "IMAGE"),
        ]
    )


def append_kontext_stage(
    workflow: Workflow,
    *,
    stage_name: str,
    source_image_1: NodeOutputRef,
    source_image_2: NodeOutputRef,
    prompt: str,
    output_prefix: str,
    position: tuple[float, float],
    seed: int = 0,
    preview: bool = True,
    save: bool = True,
) -> ChangeReport:
    if not preview and not save:
        raise ValueError("A stage must create at least one preview or save output")

    template_node, template_definition = _template_generation_node(workflow)
    generation_id = workflow.next_node_id()
    stage_uuid = str(
        uuid.uuid5(
            uuid.NAMESPACE_URL,
            f"the-foundry-cfp:{workflow.data.get('id')}:{stage_name}:{generation_id}",
        )
    )
    definition = copy.deepcopy(template_definition)
    definition["id"] = stage_uuid
    definition["name"] = stage_name
    workflow.subgraphs.append(definition)

    generation = copy.deepcopy(template_node)
    generation["id"] = generation_id
    generation["type"] = stage_uuid
    generation["title"] = stage_name
    generation["pos"] = [float(position[0]), float(position[1])]
    for item in generation.get("inputs", []):
        item["link"] = None
    for item in generation.get("outputs", []):
        item["links"] = []
    generation["widgets_values"] = [prompt, seed]

    max_order = max((node.get("order", 0) for node in workflow.nodes), default=0)
    generation["order"] = max_order + 1
    workflow.nodes.append(generation)

    next_link = workflow.next_link_id()
    _append_source_link(
        workflow, source_image_1, generation, target_slot=0, link_id=next_link
    )
    _append_source_link(
        workflow, source_image_2, generation, target_slot=1, link_id=next_link + 1
    )
    next_link += 2

    added_nodes = [generation]
    if preview:
        preview_node = copy.deepcopy(_template_node(workflow, "PreviewImage"))
        preview_node["id"] = generation_id + len(added_nodes)
        preview_node["title"] = f"Preview Candidate - {stage_name}"
        preview_node["pos"] = [float(position[0] + 825), float(position[1] + 255)]
        preview_node["order"] = max_order + len(added_nodes) + 1
        preview_node["inputs"][0]["link"] = None
        for item in preview_node.get("outputs", []):
            item["links"] = []
        workflow.nodes.append(preview_node)
        _append_node_link(
            workflow, generation, 0, preview_node, 0, next_link
        )
        next_link += 1
        added_nodes.append(preview_node)

    if save:
        save_node = copy.deepcopy(_template_node(workflow, "SaveImage"))
        save_node["id"] = generation_id + len(added_nodes)
        save_node["title"] = f"Save Candidate - {stage_name}"
        save_node["pos"] = [float(position[0] + 1245), float(position[1] + 115)]
        save_node["order"] = max_order + len(added_nodes) + 1
        save_node["inputs"][0]["link"] = None
        for item in save_node.get("outputs", []):
            item["links"] = None
        save_node["widgets_values"] = [output_prefix]
        workflow.nodes.append(save_node)
        _append_node_link(workflow, generation, 0, save_node, 0, next_link)
        next_link += 1
        added_nodes.append(save_node)

    group_id = workflow.next_group_id()
    workflow.groups.extend(
        [
            {
                "id": group_id,
                "title": f"[CFP-03 — GENERATION] {stage_name}",
                "bounding": [
                    float(position[0] - 215),
                    float(position[1] - 120),
                    815.0,
                    898.0,
                ],
                "color": "#3f789e",
                "flags": {},
            },
            {
                "id": group_id + 1,
                "title": f"[CFP-03 — OUTPUT] {stage_name}",
                "bounding": [
                    float(position[0] + 758),
                    float(position[1] - 99),
                    1146.0,
                    932.0,
                ],
                "color": "#a1309b",
                "flags": {},
            },
        ]
    )

    workflow.data["last_node_id"] = max(node["id"] for node in workflow.nodes)
    workflow.data["last_link_id"] = max(link[0] for link in workflow.links)

    first_new_link = next_link - (2 + int(preview) + int(save))
    last_new_link = next_link - 1
    return ChangeReport(
        added=[
            f"Subgraph: {stage_name} ({stage_uuid})",
            *(
                f"Node {node['id']}: {node.get('title', node.get('type'))}"
                for node in added_nodes
            ),
            f"Links {first_new_link} through {last_new_link}",
            f"Groups {group_id} and {group_id + 1}",
        ],
        updated=[
            f"Source node {source_image_1.node_id} output links",
            f"Source node {source_image_2.node_id} output links",
            "last_node_id",
            "last_link_id",
        ],
        details={
            "subgraph_id": stage_uuid,
            "node_ids": [node["id"] for node in added_nodes],
            "link_ids": list(range(first_new_link, last_new_link + 1)),
            "group_ids": [group_id, group_id + 1],
        },
    )

