from cfp.builders import append_part_isolation_stage
from cfp.models import NodeOutputRef, PartIsolationStage


def test_appending_two_stages_keeps_all_ids_unique(cfp02) -> None:
    workflow = cfp02.clone()
    first = PartIsolationStage(
        part_id="faceplate",
        display_name="Helmet Faceplate",
        source_image_1=NodeOutputRef(191),
        source_image_2=NodeOutputRef(227),
        output_prefix="CFP-03/faceplate_candidate",
    )
    second = PartIsolationStage(
        part_id="shoulder",
        display_name="Left Shoulder Armor",
        source_image_1=NodeOutputRef(191),
        source_image_2=NodeOutputRef(227),
        output_prefix="CFP-03/left_shoulder_candidate",
        position=(9700.0, 500.0),
    )

    append_part_isolation_stage(workflow, first)
    append_part_isolation_stage(workflow, second)

    node_ids = [node["id"] for node in workflow.nodes]
    link_ids = [link[0] for link in workflow.links]
    subgraph_ids = [item["id"] for item in workflow.subgraphs]
    assert len(node_ids) == len(set(node_ids))
    assert len(link_ids) == len(set(link_ids))
    assert len(subgraph_ids) == len(set(subgraph_ids))

