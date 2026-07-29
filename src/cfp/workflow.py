from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class Workflow:
    data: dict[str, Any]
    source_path: Path | None = None

    @classmethod
    def load(cls, path: str | Path) -> Workflow:
        source = Path(path)
        with source.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, dict):
            raise ValueError(f"Workflow root must be an object: {source}")
        return cls(data=data, source_path=source)

    def clone(self) -> Workflow:
        return Workflow(copy.deepcopy(self.data), self.source_path)

    def save(self, path: str | Path, *, pretty: bool = False) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8", newline="\n") as handle:
            if pretty:
                json.dump(self.data, handle, ensure_ascii=False, indent=2)
            else:
                json.dump(
                    self.data,
                    handle,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            handle.write("\n")
        return destination

    @property
    def nodes(self) -> list[dict[str, Any]]:
        return self.data.setdefault("nodes", [])

    @property
    def links(self) -> list[list[Any]]:
        return self.data.setdefault("links", [])

    @property
    def groups(self) -> list[dict[str, Any]]:
        return self.data.setdefault("groups", [])

    @property
    def subgraphs(self) -> list[dict[str, Any]]:
        definitions = self.data.setdefault("definitions", {})
        return definitions.setdefault("subgraphs", [])

    def node(self, node_id: int) -> dict[str, Any]:
        for node in self.nodes:
            if node.get("id") == node_id:
                return node
        raise KeyError(f"Node {node_id} does not exist")

    def subgraph(self, subgraph_id: str) -> dict[str, Any]:
        for subgraph in self.subgraphs:
            if subgraph.get("id") == subgraph_id:
                return subgraph
        raise KeyError(f"Subgraph {subgraph_id} does not exist")

    def next_node_id(self) -> int:
        maximum = max((node.get("id", 0) for node in self.nodes), default=0)
        return max(int(self.data.get("last_node_id", 0)), maximum) + 1

    def next_link_id(self) -> int:
        maximum = max((link[0] for link in self.links if link), default=0)
        return max(int(self.data.get("last_link_id", 0)), maximum) + 1

    def next_group_id(self) -> int:
        return max((group.get("id", 0) for group in self.groups), default=0) + 1

    def summary(self) -> dict[str, Any]:
        return {
            "workflow_id": self.data.get("id"),
            "version": self.data.get("version"),
            "revision": self.data.get("revision"),
            "nodes": len(self.nodes),
            "links": len(self.links),
            "groups": len(self.groups),
            "subgraphs": len(self.subgraphs),
            "last_node_id": self.data.get("last_node_id"),
            "last_link_id": self.data.get("last_link_id"),
        }

