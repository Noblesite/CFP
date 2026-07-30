async def comfy_entrypoint():
    from comfy_api.latest import ComfyExtension

    from .nodes import Trellis2MLXImageTo3D, Trellis2MLXModel

    class Trellis2MLXExtension(ComfyExtension):
        async def get_node_list(self):
            return [Trellis2MLXModel, Trellis2MLXImageTo3D]

    return Trellis2MLXExtension()


__all__ = ["comfy_entrypoint"]
