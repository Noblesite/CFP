from __future__ import annotations

import copy
import uuid
from collections.abc import Mapping
from typing import Any

from cfp.models import ChangeReport, NodeOutputRef
from cfp.workflow import Workflow

LEGACY_SWAN_PROMPT = (
    "Using this elegant style, create a portrait of a swan wearing a pearl "
    "tiara and lace collar, maintaining the same refined quality and soft "
    "color tones."
)


def _widget_value(node: dict[str, Any], input_name: str) -> Any | None:
    """Return the serialized widget value associated with a named node input."""
    widget_index = 0
    values = node.get("widgets_values", [])
    for item in node.get("inputs", []):
        if not isinstance(item.get("widget"), dict):
            continue
        if item.get("name") == input_name:
            return values[widget_index] if widget_index < len(values) else None
        widget_index += 1
    return None


def _set_widget_value(
    node: dict[str, Any],
    input_name: str,
    value: Any,
) -> bool:
    """Set a named widget value while preserving every unrelated widget."""
    widget_index = 0
    values = node.setdefault("widgets_values", [])
    for item in node.get("inputs", []):
        if not isinstance(item.get("widget"), dict):
            continue
        if item.get("name") == input_name:
            while len(values) <= widget_index:
                values.append(None)
            changed = values[widget_index] != value
            values[widget_index] = value
            return changed
        widget_index += 1
    return False


def _set_subgraph_input_widget(
    definition: dict[str, Any],
    input_name: str,
    value: Any,
) -> bool:
    """Synchronize the fallback widget reached by a subgraph input link."""
    inputs = definition.get("inputs", [])
    input_slot = next(
        (index for index, item in enumerate(inputs) if item.get("name") == input_name),
        None,
    )
    if input_slot is None:
        return False

    input_node_id = definition.get("inputNode", {}).get("id", -10)
    target: tuple[int, int] | None = None
    for link in definition.get("links", []):
        if isinstance(link, dict):
            if (
                link.get("origin_id") == input_node_id
                and link.get("origin_slot") == input_slot
            ):
                target = (link.get("target_id"), link.get("target_slot"))
                break
        elif (
            isinstance(link, list)
            and len(link) >= 6
            and link[1] == input_node_id
            and link[2] == input_slot
        ):
            target = (link[3], link[4])
            break

    if target is None:
        return False
    target_id, target_slot = target
    target_node = next(
        (node for node in definition.get("nodes", []) if node.get("id") == target_id),
        None,
    )
    if target_node is None:
        return False
    target_inputs = target_node.get("inputs", [])
    if not isinstance(target_slot, int) or target_slot >= len(target_inputs):
        return False
    return _set_widget_value(
        target_node,
        target_inputs[target_slot].get("name", input_name),
        value,
    )


def _remove_broken_text_quarantine(node: dict[str, Any]) -> bool:
    """Remove the obsolete proxy entry that restores the template prompt."""
    properties = node.get("properties")
    if not isinstance(properties, dict):
        return False
    quarantine = properties.get("proxyWidgetErrorQuarantine")
    if not isinstance(quarantine, list):
        return False

    kept = [
        entry
        for entry in quarantine
        if not (
            isinstance(entry, dict)
            and isinstance(entry.get("originalEntry"), list)
            and entry["originalEntry"]
            and entry["originalEntry"][-1] == "text"
        )
    ]
    if len(kept) == len(quarantine):
        return False
    if kept:
        properties["proxyWidgetErrorQuarantine"] = kept
    else:
        properties.pop("proxyWidgetErrorQuarantine", None)
    return True


def synchronize_kontext_prompts(
    workflow: Workflow,
    *,
    prompt_overrides: Mapping[int, str] | None = None,
) -> ChangeReport:
    """Make each Kontext subgraph's internal fallback match its outer prompt."""
    definitions = {item.get("id"): item for item in workflow.subgraphs}
    updated: list[str] = []
    unresolved: list[int] = []

    for node in workflow.nodes:
        definition = definitions.get(node.get("type"))
        if definition is None:
            continue
        input_names = [item.get("name") for item in node.get("inputs", [])]
        if input_names[:2] != ["image1", "image2"] or "text" not in input_names:
            continue

        node_id = node.get("id")
        prompt = _widget_value(node, "text")
        if (
            prompt_overrides is not None
            and isinstance(node_id, int)
            and node_id in prompt_overrides
        ):
            prompt = prompt_overrides[node_id]
        if not isinstance(prompt, str):
            continue
        if prompt == LEGACY_SWAN_PROMPT:
            if isinstance(node_id, int):
                unresolved.append(node_id)
            continue
        outer_changed = _set_widget_value(node, "text", prompt)
        changed = _set_subgraph_input_widget(definition, "text", prompt)
        changed = _remove_broken_text_quarantine(node) or changed
        changed = outer_changed or changed
        if changed:
            updated.append(
                f"Node {node.get('id')}: synchronized Kontext text prompt"
            )

    return ChangeReport(
        updated=updated,
        details={"unresolved_template_prompt_nodes": unresolved},
    )


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

    synchronize_kontext_prompts(workflow)
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
    _set_subgraph_input_widget(definition, "text", prompt)
    workflow.ensure_subgraphs().append(definition)

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
    _remove_broken_text_quarantine(generation)

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
