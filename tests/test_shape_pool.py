"""Face-shape helpers that need no Blender: the selection's shape pool and validation from its cache."""

import ast
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def load_shape_pool() -> type:
    """ShapePool from blender/select_identities.py (that module imports bpy, so exec only the class)."""
    tree = ast.parse((ROOT / "blender" / "select_identities.py").read_text(encoding="utf-8"))
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "ShapePool")
    namespace = {"np": np}
    exec(compile(ast.Module([node], []), "ShapePool", "exec"), namespace)
    return namespace["ShapePool"]


def rms(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(((a - b) ** 2).sum(axis=1).mean()))


def test_shape_pool_distances_match_direct_rms() -> None:
    pool_cls = load_shape_pool()
    rng = np.random.default_rng(0)
    earlier = [rng.normal(size=(30, 3)).astype(np.float32) for _ in range(5)]
    pool = pool_cls(earlier)
    added = [rng.normal(size=(30, 3)).astype(np.float32) for _ in range(140)]  # forces several growths
    for shape in added:
        pool.add(shape)
    probe = rng.normal(size=(30, 3)).astype(np.float32)
    dist = pool.distances(probe, block=16)
    everyone = earlier + added
    assert pool.n == len(everyone) and dist.shape == (len(everyone),)
    assert np.allclose(dist, [rms(s, probe) for s in everyone], atol=1e-5)
    assert np.array_equal(pool.data[5:pool.n], np.stack(added))


def test_shape_pool_starts_empty() -> None:
    pool = load_shape_pool()([])
    shape = np.ones((4, 3), np.float32)
    pool.add(shape)
    assert pool.n == 1 and np.allclose(pool.distances(shape), [0.0])


def test_validate_job_shapes_from_selection(tmp_path: Path) -> None:
    sys.path.insert(0, str(ROOT / "tools"))
    import validate_job

    rng = np.random.default_rng(1)
    shapes = rng.normal(size=(3, 10, 3)).astype(np.float32)
    cache = tmp_path / "sel.shapes.npz"
    np.savez_compressed(cache, shapes=shapes, indices=np.array([4, 9, 12]), region=np.arange(10))
    out = validate_job.shapes_from_selection(cache, ["id_00012", "id_00004"])
    assert out["identities"] == ["id_00012", "id_00004"]
    d = out["rms_distance"]
    assert d[0][0] == 0 and abs(d[0][1] - rms(shapes[2], shapes[0])) < 1e-5 and d[0][1] == d[1][0]


def test_colab_notebook_uses_the_shared_drive_folder() -> None:
    nb = json.loads((ROOT / "colab" / "synthface_3d_colab.ipynb").read_text(encoding="utf-8"))
    source = "".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    assert "1UhybtdRMuQroKlNt8RBiXOGVAkbDuIll" in source
    assert "--max-identities" in source and "--stop-at" in source and "--against" in source
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            ast.parse("".join(l for l in cell["source"] if not l.lstrip().startswith("!")))
