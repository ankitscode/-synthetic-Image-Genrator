"""Colab notebooks are committed without outputs (renders and logs belong in output/, not git)."""

import json
from pathlib import Path

import pytest

NOTEBOOKS = sorted((Path(__file__).parent.parent / "colab").glob("*.ipynb"))


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_has_no_outputs(path: Path) -> None:
    cells = json.loads(path.read_text(encoding="utf-8"))["cells"]
    dirty = [i for i, c in enumerate(cells) if c["cell_type"] == "code" and c.get("outputs")]
    assert not dirty, f"{path.name}: clear outputs (Edit > Clear all outputs) in cells {dirty}"
