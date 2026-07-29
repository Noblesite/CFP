from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class Severity(StrEnum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass(frozen=True)
class Finding:
    severity: Severity
    code: str
    message: str
    location: str | None = None

    def as_dict(self) -> dict[str, str]:
        result = {
            "severity": self.severity.value,
            "code": self.code,
            "message": self.message,
        }
        if self.location is not None:
            result["location"] = self.location
        return result


@dataclass
class ValidationReport:
    findings: list[Finding] = field(default_factory=list)

    @property
    def errors(self) -> list[Finding]:
        return [item for item in self.findings if item.severity == Severity.ERROR]

    @property
    def warnings(self) -> list[Finding]:
        return [item for item in self.findings if item.severity == Severity.WARNING]

    @property
    def is_valid(self) -> bool:
        return not self.errors

    def add(
        self,
        severity: Severity,
        code: str,
        message: str,
        location: str | None = None,
    ) -> None:
        self.findings.append(Finding(severity, code, message, location))


@dataclass(frozen=True)
class NodeOutputRef:
    node_id: int
    output_slot: int = 0


@dataclass(frozen=True)
class PartIsolationStage:
    part_id: str
    display_name: str
    source_image_1: NodeOutputRef
    source_image_2: NodeOutputRef
    source_1_identity: str = "Camera Azimuth 000 degrees"
    source_2_identity: str = "Camera Azimuth 090 degrees"
    output_prefix: str = "CFP-03/part_candidate"
    seed: int = 0
    position: tuple[float, float] = (7600.0, 500.0)


@dataclass
class ChangeReport:
    added: list[str] = field(default_factory=list)
    updated: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)

    def render(self) -> str:
        lines: list[str] = []
        if self.added:
            lines.append("Added:")
            lines.extend(f"- {item}" for item in self.added)
        if self.updated:
            lines.append("Updated:")
            lines.extend(f"- {item}" for item in self.updated)
        return "\n".join(lines)

