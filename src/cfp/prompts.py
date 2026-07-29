from __future__ import annotations

from importlib.resources import files
from string import Template


def render_prompt(template_name: str, **values: object) -> str:
    template_path = files("cfp").joinpath("prompts", template_name)
    template = Template(template_path.read_text(encoding="utf-8"))
    return template.substitute({key: str(value) for key, value in values.items()})

