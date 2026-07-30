async def comfy_entrypoint():
    from comfy_api.latest import ComfyExtension

    from .nodes import Trellis2MLXImageTo3D, Trellis2MLXMeshReport, Trellis2MLXModel

    class Trellis2MLXExtension(ComfyExtension):
        async def get_node_list(self):
            return [Trellis2MLXModel, Trellis2MLXImageTo3D, Trellis2MLXMeshReport]

    return Trellis2MLXExtension()


WEB_DIRECTORY = "./web"

__all__ = ["WEB_DIRECTORY", "comfy_entrypoint"]
