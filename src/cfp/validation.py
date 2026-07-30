from __future__ import annotations

import re
from collections import Counter
from typing import Any, Iterable

from cfp.models import Severity, ValidationReport
from cfp.workflow import Workflow

UUID_TYPE = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)


def _duplicates(values: Iterable[Any]) -> list[Any]:
    return [value for value, count in Counter(values).items() if count > 1]


def _slot_exists(node: dict[str, Any], field: str, slot: Any) -> bool:
    return (
        isinstance(slot, int)
        and slot >= 0
        and isinstance(node.get(field), list)
        and slot < len(node[field])
    )


def validate_workflow(workflow: Workflow) -> ValidationReport:
    report = ValidationReport()
    data = workflow.data

    required = ("nodes", "links", "groups", "last_node_id", "last_link_id")
    for field in required:
        if field not in data:
            report.add(
                Severity.ERROR,
                "missing_required_field",
                f"Required top-level field '{field}' is missing",
                field,
            )

    nodes = data.get("nodes", [])
    links = data.get("links", [])
    if not isinstance(nodes, list) or not isinstance(links, list):
        report.add(
            Severity.ERROR,
            "invalid_graph_container",
            "Top-level nodes and links must be arrays",
        )
        return report

    node_ids = [node.get("id") for node in nodes]
    link_ids = [link[0] for link in links if isinstance(link, list) and link]
    for node_id in _duplicates(node_ids):
        report.add(
            Severity.ERROR,
            "duplicate_node_id",
            f"Top-level node ID {node_id} is duplicated",
            f"nodes[{node_id}]",
        )
    for link_id in _duplicates(link_ids):
        report.add(
            Severity.ERROR,
            "duplicate_link_id",
            f"Top-level link ID {link_id} is duplicated",
            f"links[{link_id}]",
        )

    node_by_id = {node.get("id"): node for node in nodes}
    link_by_id: dict[int, list[Any]] = {}
    for index, link in enumerate(links):
        if not isinstance(link, list) or len(link) < 6:
            report.add(
                Severity.ERROR,
                "invalid_link_shape",
                "Top-level links must contain at least six values",
                f"links[{index}]",
            )
            continue
        link_id, origin_id, origin_slot, target_id, target_slot, _ = link[:6]
        link_by_id[link_id] = link
        origin = node_by_id.get(origin_id)
        target = node_by_id.get(target_id)
        if origin is None:
            report.add(
                Severity.ERROR,
                "missing_origin_node",
                f"Link {link_id} references missing origin node {origin_id}",
                f"links[{link_id}]",
            )
        elif not _slot_exists(origin, "outputs", origin_slot):
            report.add(
                Severity.ERROR,
                "invalid_origin_slot",
                f"Link {link_id} references invalid output slot {origin_slot} on node {origin_id}",
                f"links[{link_id}]",
            )
        if target is None:
            report.add(
                Severity.ERROR,
                "missing_target_node",
                f"Link {link_id} references missing target node {target_id}",
                f"links[{link_id}]",
            )
        elif not _slot_exists(target, "inputs", target_slot):
            report.add(
                Severity.ERROR,
                "invalid_target_slot",
                f"Link {link_id} references invalid input slot {target_slot} on node {target_id}",
                f"links[{link_id}]",
            )

    for node in nodes:
        node_id = node.get("id")
        for slot, item in enumerate(node.get("inputs", [])):
            link_id = item.get("link")
            if link_id is None:
                continue
            link = link_by_id.get(link_id)
            if link is None:
                report.add(
                    Severity.ERROR,
                    "missing_input_link",
                    f"Node {node_id} input {slot} references missing link {link_id}",
                    f"nodes[{node_id}].inputs[{slot}]",
                )
            elif link[3] != node_id or link[4] != slot:
                report.add(
                    Severity.ERROR,
                    "input_link_mismatch",
                    f"Node {node_id} input {slot} does not match link {link_id}'s target",
                    f"nodes[{node_id}].inputs[{slot}]",
                )
        for slot, item in enumerate(node.get("outputs", [])):
            for link_id in item.get("links") or []:
                link = link_by_id.get(link_id)
                if link is None:
                    report.add(
                        Severity.ERROR,
                        "missing_output_link",
                        f"Node {node_id} output {slot} references missing link {link_id}",
                        f"nodes[{node_id}].outputs[{slot}]",
                    )
                elif link[1] != node_id or link[2] != slot:
                    report.add(
                        Severity.ERROR,
                        "output_link_mismatch",
                        f"Node {node_id} output {slot} does not match link {link_id}'s origin",
                        f"nodes[{node_id}].outputs[{slot}]",
                    )

    max_node_id = max((value for value in node_ids if isinstance(value, int)), default=0)
    max_link_id = max((value for value in link_ids if isinstance(value, int)), default=0)
    if isinstance(data.get("last_node_id"), int) and data["last_node_id"] < max_node_id:
        report.add(
            Severity.ERROR,
            "last_node_id_too_low",
            f"last_node_id {data['last_node_id']} is lower than node ID {max_node_id}",
            "last_node_id",
        )
    if isinstance(data.get("last_link_id"), int) and data["last_link_id"] < max_link_id:
        report.add(
            Severity.ERROR,
            "last_link_id_too_low",
            f"last_link_id {data['last_link_id']} is lower than link ID {max_link_id}",
            "last_link_id",
        )

    subgraphs = data.get("definitions", {}).get("subgraphs", [])
    subgraph_ids = [subgraph.get("id") for subgraph in subgraphs]
    for subgraph_id in _duplicates(subgraph_ids):
        report.add(
            Severity.ERROR,
            "duplicate_subgraph_id",
            f"Subgraph definition ID {subgraph_id} is duplicated",
            "definitions.subgraphs",
        )
    known_subgraphs = set(subgraph_ids)
    for node in nodes:
        node_type = node.get("type")
        if (
            isinstance(node_type, str)
            and UUID_TYPE.fullmatch(node_type)
            and node_type not in known_subgraphs
        ):
            report.add(
                Severity.ERROR,
                "missing_subgraph_definition",
                f"Node {node.get('id')} references missing subgraph {node_type}",
                f"nodes[{node.get('id')}].type",
            )

    for subgraph in subgraphs:
        _validate_subgraph(subgraph, report)
    return report


def _validate_subgraph(
    subgraph: dict[str, Any],
    report: ValidationReport,
) -> None:
    name = subgraph.get("name", subgraph.get("id", "<unknown>"))
    nodes = subgraph.get("nodes", [])
    links = subgraph.get("links", [])
    node_ids = [node.get("id") for node in nodes]
    link_ids = [link.get("id") for link in links if isinstance(link, dict)]
    for node_id in _duplicates(node_ids):
        report.add(
            Severity.ERROR,
            "duplicate_internal_node_id",
            f"Subgraph '{name}' has duplicate internal node ID {node_id}",
            f"subgraphs[{name}].nodes",
        )
    for link_id in _duplicates(link_ids):
        report.add(
            Severity.ERROR,
            "duplicate_internal_link_id",
            f"Subgraph '{name}' has duplicate internal link ID {link_id}",
            f"subgraphs[{name}].links",
        )

    node_by_id = {node.get("id"): node for node in nodes}
    valid_ids = set(node_by_id) | {
        subgraph.get("inputNode", {}).get("id", -10),
        subgraph.get("outputNode", {}).get("id", -20),
    }
    for index, link in enumerate(links):
        if not isinstance(link, dict):
            report.add(
                Severity.ERROR,
                "invalid_internal_link_shape",
                f"Subgraph '{name}' internal link {index} is not an object",
                f"subgraphs[{name}].links[{index}]",
            )
            continue
        origin_id = link.get("origin_id")
        target_id = link.get("target_id")
        if origin_id not in valid_ids:
            report.add(
                Severity.ERROR,
                "missing_internal_origin",
                f"Subgraph '{name}' link {link.get('id')} references missing origin {origin_id}",
                f"subgraphs[{name}].links[{index}]",
            )
        if target_id not in valid_ids:
            report.add(
                Severity.ERROR,
                "missing_internal_target",
                f"Subgraph '{name}' link {link.get('id')} references missing target {target_id}",
                f"subgraphs[{name}].links[{index}]",
            )
        origin = node_by_id.get(origin_id)
        target = node_by_id.get(target_id)
        if origin is not None and not _slot_exists(origin, "outputs", link.get("origin_slot")):
            report.add(
                Severity.ERROR,
                "invalid_internal_origin_slot",
                f"Subgraph '{name}' link {link.get('id')} has invalid origin slot",
                f"subgraphs[{name}].links[{index}]",
            )
        if target is not None and not _slot_exists(target, "inputs", link.get("target_slot")):
            report.add(
                Severity.ERROR,
                "invalid_internal_target_slot",
                f"Subgraph '{name}' link {link.get('id')} has invalid target slot",
                f"subgraphs[{name}].links[{index}]",
            )
