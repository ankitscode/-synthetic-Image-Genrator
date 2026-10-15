"""5 landmarks from the 3D model (no detector): eye centres from the Eyes mesh, nose tip and
mouth corners from basemesh vertices (config/landmark_vertices.json).

Order (viewer's point of view, as in the ArcFace template): left eye, right eye, nose tip,
left mouth corner, right mouth corner. The camera looks at the face from the front (-Y), so the
viewer's left is the subject's right.
"""

import json
from pathlib import Path

import bpy
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Vector

import human

LANDMARK_NAMES = ["left_eye", "right_eye", "nose_tip", "left_mouth", "right_mouth"]  # viewer's view
VERTICES_FILE = Path(__file__).resolve().parent.parent / "config" / "landmark_vertices.json"
VISIBLE_TOLERANCE_M = 0.003  # first ray hit within 3 mm of the landmark = visible
RAY_STEP_M = 1e-4
DEBUG_DISTANCES = False


def load_vertices() -> dict[str, int]:
    return {k: v for k, v in json.loads(VERTICES_FILE.read_text(encoding="utf-8")).items() if k != "comment"}


def world_points(basemesh, eyes, vertices: dict[str, int]) -> dict[str, Vector]:
    """World positions of the 5 landmarks, keyed by LANDMARK_NAMES (viewer's left/right)."""
    subject_left_eye, subject_right_eye = human.eye_centres(eyes)
    co = human._world_coords(basemesh)
    return {
        "left_eye": subject_right_eye,
        "right_eye": subject_left_eye,
        "nose_tip": Vector(co[vertices["nose_tip"]]),
        "left_mouth": Vector(co[vertices["mouth_subject_right"]]),
        "right_mouth": Vector(co[vertices["mouth_subject_left"]]),
    }


def _visible(camera, point: Vector, target, on_surface: bool, see_through: set[str]) -> tuple[bool, str | None]:
    """Ray from the camera to the point. Visible when the first solid hit is `target` (eyelash /
    eyebrow cards are stepped through); for a point on the skin (on_surface) the hit must also be
    at the point. Eye centres sit inside the eyeball, so any hit on the eye mesh counts.
    Hair particles (beards) are not ray-cast, so a beard never hides a point here."""
    scene = bpy.context.scene
    depsgraph = bpy.context.evaluated_depsgraph_get()
    origin = camera.matrix_world.translation.copy()
    direction = (point - origin).normalized()
    for _ in range(8):
        remaining = (point - origin).length + VISIBLE_TOLERANCE_M
        hit, location, _, _, obj, _ = scene.ray_cast(depsgraph, origin, direction, distance=remaining)
        if not hit:
            return True, None
        if obj.original.name in see_through:
            origin = location + direction * RAY_STEP_M
            continue
        if obj.original == target and (not on_surface or (location - point).length <= VISIBLE_TOLERANCE_M):
            return True, None
        if DEBUG_DISTANCES:
            print(f"BLOCK_DIST {obj.original.name} {(location - point).length * 1000:.1f}mm")
        return False, obj.original.name
    return False, "too many see-through layers"


def project(camera, points: dict[str, Vector], basemesh, eyes, see_through: set[str]) -> dict:
    """Pixel coordinates (origin top-left) + visibility for the current camera and resolution."""
    scene = bpy.context.scene
    pct = scene.render.resolution_percentage / 100.0
    width, height = scene.render.resolution_x * pct, scene.render.resolution_y * pct
    out = {"points": [], "visible": [], "blocked_by": []}
    for name in LANDMARK_NAMES:
        point = points[name]
        ndc = world_to_camera_view(scene, camera, point)
        out["points"].append([round(ndc.x * width, 3), round((1.0 - ndc.y) * height, 3)])
        is_eye = name.endswith("eye")
        visible, blocked = _visible(camera, point, eyes if is_eye else basemesh, not is_eye, see_through)
        out["visible"].append(visible)
        out["blocked_by"].append(blocked)
    return out
