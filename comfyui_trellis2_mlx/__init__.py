async def comfy_entrypoint():
    from comfy_api.latest import ComfyExtension

    from .nodes import (
        Trellis2MLXInputMaskQualityGate,
        Trellis2MLXModelSheetConsistencyGate,
        Trellis2MLXModelSheetAlignmentReview,
        Trellis2MLXModelSheetAlignmentCandidate,
        Trellis2MLXImageTo3D,
        Trellis2MLXImageConditioning,
        Trellis2MLXMeshReport,
        Trellis2MLXModel,
        Trellis2MLXMultiViewTo3D,
        Trellis2MLXRemoveFloaters,
        Trellis2MLXTopologyDiagnostics,
        Trellis2MLXTopologySanitizer,
        Trellis2MLXBackgroundGeometryGuard,
        Trellis2MLXPostVoxelTopologyPolish,
        Trellis2MLXPrintScaleFeatureGate,
        Trellis2MLXVoxelRemeshCandidate,
        Trellis2MLXVoxelCandidateComparison,
    )

    class Trellis2MLXExtension(ComfyExtension):
        async def get_node_list(self):
            return [
                Trellis2MLXInputMaskQualityGate,
                Trellis2MLXModelSheetConsistencyGate,
                Trellis2MLXModelSheetAlignmentReview,
                Trellis2MLXModelSheetAlignmentCandidate,
                Trellis2MLXModel,
                Trellis2MLXImageConditioning,
                Trellis2MLXImageTo3D,
                Trellis2MLXMultiViewTo3D,
                Trellis2MLXMeshReport,
                Trellis2MLXRemoveFloaters,
                Trellis2MLXTopologyDiagnostics,
                Trellis2MLXTopologySanitizer,
                Trellis2MLXBackgroundGeometryGuard,
                Trellis2MLXVoxelRemeshCandidate,
                Trellis2MLXPostVoxelTopologyPolish,
                Trellis2MLXPrintScaleFeatureGate,
                Trellis2MLXVoxelCandidateComparison,
            ]

    return Trellis2MLXExtension()


WEB_DIRECTORY = "./web"

__all__ = ["WEB_DIRECTORY", "comfy_entrypoint"]
