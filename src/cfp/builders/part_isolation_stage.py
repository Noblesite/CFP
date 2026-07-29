from __future__ import annotations

from cfp.builders.kontext_stage import append_kontext_stage
from cfp.models import ChangeReport, PartIsolationStage
from cfp.prompts import render_prompt
from cfp.workflow import Workflow


def append_part_isolation_stage(
    workflow: Workflow,
    stage: PartIsolationStage,
) -> ChangeReport:
    prompt = render_prompt(
        "part_isolation.txt",
        part_name=stage.display_name,
        source_1_identity=stage.source_1_identity,
        source_2_identity=stage.source_2_identity,
    )
    return append_kontext_stage(
        workflow,
        stage_name=f"Isolate {stage.display_name}",
        source_image_1=stage.source_image_1,
        source_image_2=stage.source_image_2,
        prompt=prompt,
        output_prefix=stage.output_prefix,
        position=stage.position,
        seed=stage.seed,
    )

