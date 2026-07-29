from pathlib import Path

import pytest

from cfp.workflow import Workflow

FIXTURE_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def cfp02_path() -> Path:
    return FIXTURE_DIR / "CFP-02.json"


@pytest.fixture
def cfp02(cfp02_path: Path) -> Workflow:
    return Workflow.load(cfp02_path)

