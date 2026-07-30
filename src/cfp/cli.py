from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from cfp.builders import append_part_isolation_stage, synchronize_kontext_prompts
from cfp.models import NodeOutputRef, PartIsolationStage, Severity
from cfp.validation import validate_workflow
from cfp.workflow import Workflow


def _print_validation(report: object) -> None:
    findings = getattr(report, "findings")
    if not findings:
        print("Validation passed with no findings.")
        return
    for finding in findings:
        location = f" [{finding.location}]" if finding.location else ""
        print(
            f"{finding.severity.value.upper()} {finding.code}{location}: "
            f"{finding.message}"
        )
    print(
        f"{len(getattr(report, 'errors'))} error(s), "
        f"{len(getattr(report, 'warnings'))} warning(s)"
    )


def _cmd_validate(args: argparse.Namespace) -> int:
    workflow = Workflow.load(args.workflow)
    report = validate_workflow(workflow)
    if args.json:
        print(json.dumps([item.as_dict() for item in report.findings], indent=2))
    else:
        _print_validation(report)
    if report.errors or (args.strict and report.warnings):
        return 1
    return 0


def _cmd_inspect(args: argparse.Namespace) -> int:
    workflow = Workflow.load(args.workflow)
    summary = workflow.summary()
    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        for key, value in summary.items():
            print(f"{key}: {value}")
        print("stages:")
        definitions = {item.get("id"): item for item in workflow.subgraphs}
        for node in workflow.nodes:
            if node.get("type") in definitions:
                print(f"- node {node['id']}: {node.get('title', definitions[node['type']].get('name'))}")
    return 0


def _cmd_append_part_isolation(args: argparse.Namespace) -> int:
    source_path = Path(args.workflow)
    destination = source_path if args.in_place else Path(args.output)
    workflow = Workflow.load(source_path)
    before = validate_workflow(workflow)
    if before.errors:
        print("Source workflow has validation errors; refusing to edit.", file=sys.stderr)
        _print_validation(before)
        return 2

    stage = PartIsolationStage(
        part_id=args.part_id,
        display_name=args.part_name,
        source_image_1=NodeOutputRef(args.source_1_node, args.source_1_slot),
        source_image_2=NodeOutputRef(args.source_2_node, args.source_2_slot),
        source_1_identity=args.source_1_identity,
        source_2_identity=args.source_2_identity,
        output_prefix=args.output_prefix,
        seed=args.seed,
        position=(args.position_x, args.position_y),
    )
    change_report = append_part_isolation_stage(workflow, stage)
    after = validate_workflow(workflow)
    if after.errors or (args.strict and after.warnings):
        print("Edit introduced validation findings; refusing to save.", file=sys.stderr)
        _print_validation(after)
        return 3

    workflow.save(destination, pretty=args.pretty)
    print(change_report.render())
    print(f"Saved: {destination}")
    return 0


def _cmd_repair_kontext_prompts(args: argparse.Namespace) -> int:
    source_path = Path(args.workflow)
    destination = source_path if args.in_place else Path(args.output)
    workflow = Workflow.load(source_path)
    before = validate_workflow(workflow)
    if before.errors:
        print("Source workflow has validation errors; refusing to edit.", file=sys.stderr)
        _print_validation(before)
        return 2

    prompt_overrides: dict[int, str] | None = None
    if args.prompt_source:
        prompt_source = Workflow.load(args.prompt_source)
        prompt_overrides = {}
        for node in prompt_source.nodes:
            inputs = node.get("inputs", [])
            if "text" not in [item.get("name") for item in inputs]:
                continue
            widget_values = node.get("widgets_values", [])
            if widget_values and isinstance(widget_values[0], str):
                prompt_overrides[node["id"]] = widget_values[0]
    if (args.node_id is None) != (args.prompt_file is None):
        print("--node-id and --prompt-file must be supplied together.", file=sys.stderr)
        return 2
    if args.node_id is not None:
        prompt_overrides = prompt_overrides or {}
        prompt_overrides[args.node_id] = Path(args.prompt_file).read_text(
            encoding="utf-8"
        ).strip()

    change_report = synchronize_kontext_prompts(
        workflow,
        prompt_overrides=prompt_overrides,
    )
    unresolved = change_report.details.get("unresolved_template_prompt_nodes", [])
    if unresolved:
        print(
            "Cannot infer the intended prompt for Kontext node(s) "
            f"{', '.join(str(item) for item in unresolved)}; provide "
            "--prompt-source with a known-good workflow.",
            file=sys.stderr,
        )
        return 4
    after = validate_workflow(workflow)
    if after.errors or (args.strict and after.warnings):
        print("Repair introduced validation findings; refusing to save.", file=sys.stderr)
        _print_validation(after)
        return 3

    workflow.save(destination, pretty=args.pretty)
    if change_report.updated:
        print(change_report.render())
    else:
        print("No Kontext prompt repairs were required.")
    print(f"Saved: {destination}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cfp")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate", help="Validate workflow graph integrity")
    validate.add_argument("workflow")
    validate.add_argument("--strict", action="store_true")
    validate.add_argument("--json", action="store_true")
    validate.set_defaults(func=_cmd_validate)

    inspect = subparsers.add_parser("inspect", help="Summarize a workflow")
    inspect.add_argument("workflow")
    inspect.add_argument("--json", action="store_true")
    inspect.set_defaults(func=_cmd_inspect)

    append = subparsers.add_parser(
        "append-part-isolation",
        help="Append a two-reference FLUX Kontext part-isolation stage",
    )
    append.add_argument("--workflow", required=True)
    append.add_argument("--part-id", required=True)
    append.add_argument("--part-name", required=True)
    append.add_argument("--source-1-node", required=True, type=int)
    append.add_argument("--source-1-slot", type=int, default=0)
    append.add_argument("--source-2-node", required=True, type=int)
    append.add_argument("--source-2-slot", type=int, default=0)
    append.add_argument(
        "--source-1-identity",
        default="Camera Azimuth 000 degrees",
    )
    append.add_argument(
        "--source-2-identity",
        default="Camera Azimuth 090 degrees",
    )
    append.add_argument("--output-prefix", required=True)
    append.add_argument("--seed", type=int, default=0)
    append.add_argument("--position-x", type=float, default=7600.0)
    append.add_argument("--position-y", type=float, default=500.0)
    destination = append.add_mutually_exclusive_group(required=True)
    destination.add_argument("--output")
    destination.add_argument("--in-place", action="store_true")
    append.add_argument("--pretty", action="store_true")
    append.add_argument("--strict", action="store_true")
    append.set_defaults(func=_cmd_append_part_isolation)

    repair = subparsers.add_parser(
        "repair-kontext-prompts",
        help="Synchronize nested Kontext prompts and remove stale text proxies",
    )
    repair.add_argument("--workflow", required=True)
    repair.add_argument(
        "--prompt-source",
        help="Known-good workflow whose outer prompts are copied by node ID",
    )
    repair.add_argument(
        "--node-id",
        type=int,
        help="Kontext node to restore from --prompt-file",
    )
    repair.add_argument(
        "--prompt-file",
        help="UTF-8 text file containing the authoritative prompt for --node-id",
    )
    destination = repair.add_mutually_exclusive_group(required=True)
    destination.add_argument("--output")
    destination.add_argument("--in-place", action="store_true")
    repair.add_argument("--pretty", action="store_true")
    repair.add_argument("--strict", action="store_true")
    repair.set_defaults(func=_cmd_repair_kontext_prompts)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
