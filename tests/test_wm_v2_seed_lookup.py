"""Run with python tests/test_wm_v2_seed_lookup.py; no GPU dependencies needed."""

import ast
import os
from pathlib import Path
from types import SimpleNamespace


# Execute the actual loader method, stubbing only image computation and setup.
source = Path(__file__).resolve().parents[1] / "utils" / "weight_map_utils.py"
tree = ast.parse(source.read_text())
loader_class = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "WeightMapLoader")
method = next(n for n in loader_class.body if isinstance(n, ast.FunctionDef) and n.name == "_compute_wm_v2")
namespace = {
    "os": os,
    "compute_wm_v2": lambda image, **kwargs: {"wm": kwargs["colmap_seed_mask"]},
}
exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), namespace)

mask, direct_mask = object(), object()
stem = "Image__2024-05-06T074850.368734_6091"
loader = SimpleNamespace(
    _read_bgr=lambda camera: SimpleNamespace(shape=(2, 3, 3)),
    _ensure_colmap_seeds=lambda shape: None,
    _colmap_seed_masks={stem: mask, "0001": mask, "direct.jpg": direct_mask, "direct": mask},
    wm_v2_kmeans_k=16,
    wm_v2_caustic_L=200,
    wm_v2_caustic_chroma=40,
    wm_v2_caustic_dilate=9,
)
for name, expected in [
    (stem + ".jpg", mask),
    (stem, mask),
    ("0001", mask),
    ("direct.jpg", direct_mask),
    ("missing.jpg", None),
]:
    result = namespace["_compute_wm_v2"](loader, SimpleNamespace(image_name=name))
    assert result is expected, f"Incorrect COLMAP seed for {name}"
print("PASS: wm_v2 forwards seeds for filename/stem names and preserves direct matches")
