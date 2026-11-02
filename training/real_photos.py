"""Real test photos (consented, evaluation only, never training): hand-clicked landmarks and alignment.

Layout (on Drive, never in git):
    <real_dir>/<person>/<photo>.jpg     one folder per person, any number of photos
    <real_dir>/landmarks.json           {"points": {"<person>/<photo>": [[x, y] x 5] | "skip"}}

No face detector (CLAUDE.md hard rule 3): the 5 points are clicked by hand in Colab (clicker()), in the
pipeline's order: left eye centre, right eye centre, nose tip, left mouth corner, right mouth corner, left/right
as seen in the image. Coordinates are in pixels of the full-size photo after EXIF rotation.

Alignment: the photo is first shrunk with an anti-aliased resize so the eyes are as far apart as in a 256 px render,
then the same similarity transform to the ArcFace template as the renders (app/postprocess.align).
"""

from __future__ import annotations

import base64
import io
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from shared import constants as C

EXTENSIONS = {".jpg", ".jpeg", ".png"}
LANDMARKS_FILE = "landmarks.json"
SKIP = "skip"
TEMPLATE_IOD = float(np.hypot(C.ARCFACE_112[1][0] - C.ARCFACE_112[0][0], C.ARCFACE_112[1][1] - C.ARCFACE_112[0][1]))
PRESCALE_IOD = 2 * TEMPLATE_IOD     # eye distance before alignment, about that of a 256 px render
CLICK_DISPLAY_MAX = 1400            # long side of the photo sent to the click page
POINT_NAMES = ["LEFT eye centre", "RIGHT eye centre", "nose tip", "LEFT mouth corner", "RIGHT mouth corner"]


def list_photos(real_dir: Path) -> dict[str, list[str]]:
    """person -> sorted photo keys ("<person>/<file>")."""
    people = {}
    for d in sorted(p for p in real_dir.iterdir() if p.is_dir() and not p.name.startswith((".", "_"))):
        files = sorted(f.name for f in d.iterdir() if f.suffix.lower() in EXTENSIONS)
        if files:
            people[d.name] = [f"{d.name}/{f}" for f in files]
    return people


def load_photo(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(ImageOps.exif_transpose(im).convert("RGB"))


def load_landmarks(real_dir: Path) -> dict:
    path = real_dir / LANDMARKS_FILE
    return json.loads(path.read_text()) if path.exists() else {"order": POINT_NAMES, "points": {}}


def save_landmark(real_dir: Path, key: str, points: list | str) -> None:
    marks = load_landmarks(real_dir)
    marks["points"][key] = points
    tmp = real_dir / (LANDMARKS_FILE + ".tmp")
    tmp.write_text(json.dumps(marks, indent=1) + "\n")
    os.replace(tmp, real_dir / LANDMARKS_FILE)


def align_photo(image: np.ndarray, points: np.ndarray) -> tuple[np.ndarray, dict]:
    """112x112 aligned crop of a full-size photo from its 5 clicked points."""
    from app.postprocess import align   # OpenCV, as for the renders

    iod = float(np.linalg.norm(points[1] - points[0]))
    if iod <= 0:
        raise ValueError("eye points coincide")
    factor = min(1.0, PRESCALE_IOD / iod)
    if factor < 1.0:
        h, w = image.shape[:2]
        size = (max(1, round(w * factor)), max(1, round(h * factor)))
        image = np.asarray(Image.fromarray(image).resize(size, Image.Resampling.LANCZOS))
        points = points * np.array([size[0] / w, size[1] / h])
    return align(np.ascontiguousarray(image), points.astype(np.float32))


def load_aligned(real_dir: Path) -> dict:
    """Aligned crops of every clicked photo: images (N, 112, 112, 3), labels (person index), names, files,
    plus what was left out (skipped, not clicked yet, people with fewer than 2 usable photos)."""
    marks = load_landmarks(real_dir)["points"]
    images, labels, files, names, residuals = [], [], [], [], []
    missing, skipped, too_few = [], [], []
    for person, keys in list_photos(real_dir).items():
        usable = []
        for key in keys:
            pts = marks.get(key)
            if pts is None:
                missing.append(key)
            elif pts == SKIP:
                skipped.append(key)
            else:
                usable.append((key, np.array(pts, np.float64)))
        if len(usable) < 2:           # one photo cannot make a same-person pair
            too_few.append(person)
            continue
        for key, pts in usable:
            crop, info = align_photo(load_photo(real_dir / key), pts)
            images.append(crop)
            labels.append(len(names))
            files.append(Path(key).stem)
            residuals.append(info["residual_px"])
        names.append(f"real/{person}")
    return {"images": np.stack(images) if images else np.zeros((0, 112, 112, 3), np.uint8),
            "labels": np.array(labels, np.int64), "names": names, "files": files,
            "residual_px": residuals, "missing": missing, "skipped": skipped, "too_few": too_few}


def aligned_sheet(aligned: dict, path: Path, per_row: int = 8) -> None:
    """One row per person: the aligned crops, to check the clicks before trusting the numbers."""
    rows = [np.flatnonzero(aligned["labels"] == k) for k in range(len(aligned["names"]))]
    width = max([min(len(r), per_row) for r in rows] + [1])
    lines = sum(-(-len(r) // per_row) for r in rows)
    sheet = Image.new("RGB", (width * 112, max(1, lines) * 112), "white")
    y = 0
    for r in rows:
        for k, idx in enumerate(r):
            sheet.paste(Image.fromarray(aligned["images"][idx]), ((k % per_row) * 112, y + (k // per_row) * 112))
        y += -(-len(r) // per_row) * 112
    sheet.save(path)


# ---------------------------------------------------------------- click page (Colab)

CLICK_PAGE = """
<div id="sf" style="font:14px system-ui,sans-serif;color:#1d1d1b;background:#fff;padding:10px;border-radius:8px">
 <div id="sf-status" style="font-weight:600;height:1.5em;overflow:hidden;white-space:nowrap;text-overflow:ellipsis"></div>
 <div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-bottom:6px">
  <button id="sf-undo">Undo (z)</button><button id="sf-skip">Skip photo (s)</button>
  <button id="sf-prev">&larr; Previous</button><button id="sf-next">Next &rarr;</button>
  <label>zoom <select id="sf-zoom"><option>1</option><option>1.5</option><option selected>2</option>
   <option>3</option><option>4</option></select></label>
 </div>
 <div id="sf-ask" style="font-size:16px;margin:4px 0 6px;height:2.8em;overflow:hidden"></div>
 <div id="sf-box" style="max-height:720px;overflow:auto;border:1px solid #ccc"><canvas id="sf-canvas" style="cursor:crosshair"></canvas></div>
</div>
<script>
(() => {
 const KEYS = __KEYS__, NAMES = __NAMES__, COLOURS = ["#00e5ff", "#ff2bd6", "#ffd400", "#39ff14", "#ff6a00"];
 let done = __DONE__, i = __START__, pts = [], img = new Image(), scale = 1;
 const cv = document.getElementById("sf-canvas"), ctx = cv.getContext("2d"), zoomSel = document.getElementById("sf-zoom");
 const box = document.getElementById("sf-box");
 function centre(fx, fy) {  // keep this point of the photo (0..1) in the middle of the view
  box.scrollLeft = fx * cv.width - box.clientWidth / 2; box.scrollTop = fy * cv.height - box.clientHeight / 2;
 }
 const call = async (name, args) => (await google.colab.kernel.invokeFunction(name, args, {})).data["application/json"];
 function draw() {
  const z = parseFloat(zoomSel.value), fit = Math.min(1, 900 / img.width);
  cv.width = img.width * fit * z; cv.height = img.height * fit * z;
  ctx.drawImage(img, 0, 0, cv.width, cv.height);
  pts.forEach((p, k) => {
   const x = p[0] / scale * cv.width / img.width, y = p[1] / scale * cv.height / img.height;
   ctx.strokeStyle = COLOURS[k]; ctx.lineWidth = 2;
   ctx.beginPath(); ctx.arc(x, y, 6, 0, 7); ctx.stroke();
   ctx.beginPath(); ctx.moveTo(x - 12, y); ctx.lineTo(x + 12, y); ctx.moveTo(x, y - 12); ctx.lineTo(x, y + 12); ctx.stroke();
   ctx.fillStyle = COLOURS[k]; ctx.font = "bold 14px sans-serif"; ctx.fillText(String(k + 1), x + 8, y - 8);
  });
  const left = KEYS.filter(k => !(k in done)).length;
  document.getElementById("sf-status").textContent = `photo ${i + 1}/${KEYS.length} · ${KEYS[i]} · ${left} left` +
   (done[KEYS[i]] === "skip" ? " · SKIPPED" : (KEYS[i] in done ? " · saved" : ""));
  document.getElementById("sf-ask").innerHTML = pts.length < 5
   ? `Click <b style="color:${COLOURS[pts.length]};background:#222;padding:0 4px">${pts.length + 1}. ${NAMES[pts.length]}</b> (left / right as seen in the photo)`
   : "Saved. Loading the next photo...";
 }
 async function show(k) {
  i = Math.max(0, Math.min(KEYS.length - 1, k));
  const r = await call("synthface.photo", [KEYS[i]]);
  scale = r.scale; pts = Array.isArray(done[KEYS[i]]) ? done[KEYS[i]].slice() : [];
  img = new Image(); img.onload = () => { draw(); centre(0.5, 0.45); }; img.src = r.data;
 }
 function nextOpen() {
  for (let k = 1; k <= KEYS.length; k++) { const j = (i + k) % KEYS.length; if (!(KEYS[j] in done)) return show(j); }
  document.getElementById("sf-ask").innerHTML = "<b>All photos are done.</b> Run the next cell.";
 }
 cv.addEventListener("click", async e => {
  if (pts.length >= 5) pts = [];
  const b = cv.getBoundingClientRect();
  pts.push([(e.clientX - b.left) * img.width / cv.width * scale, (e.clientY - b.top) * img.height / cv.height * scale]);
  draw();
  if (pts.length === 5) { await call("synthface.save", [KEYS[i], pts]); done[KEYS[i]] = pts.slice(); setTimeout(nextOpen, 400); }
 });
 const undo = () => { pts.pop(); draw(); };
 const skip = async () => { await call("synthface.save", [KEYS[i], "skip"]); done[KEYS[i]] = "skip"; nextOpen(); };
 document.getElementById("sf-undo").onclick = undo;
 document.getElementById("sf-skip").onclick = skip;
 document.getElementById("sf-prev").onclick = () => show(i - 1);
 document.getElementById("sf-next").onclick = () => show(i + 1);
 zoomSel.onchange = () => {
  const fx = (box.scrollLeft + box.clientWidth / 2) / cv.width, fy = (box.scrollTop + box.clientHeight / 2) / cv.height;
  draw(); centre(fx, fy);
 };
 document.addEventListener("keydown", e => { if (e.key === "z") undo(); if (e.key === "s") skip(); });
 show(i);
})();
</script>
"""


def photo_for_page(real_dir: Path, key: str) -> dict:
    """Downscaled JPEG (data URL) of a photo + the factor from page pixels back to full-size pixels."""
    image = Image.fromarray(load_photo(real_dir / key))
    scale = max(1.0, max(image.size) / CLICK_DISPLAY_MAX)
    if scale > 1:
        image = image.resize((round(image.width / scale), round(image.height / scale)), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    image.save(buf, "JPEG", quality=90)
    return {"data": "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode(), "scale": scale}


def click_page(real_dir: Path) -> str:
    keys = [k for ks in list_photos(real_dir).values() for k in ks]
    if not keys:
        raise SystemExit(f"no photos in {real_dir}/<person>/")
    done = load_landmarks(real_dir)["points"]
    start = next((n for n, k in enumerate(keys) if k not in done), 0)
    return (CLICK_PAGE.replace("__KEYS__", json.dumps(keys)).replace("__NAMES__", json.dumps(POINT_NAMES))
            .replace("__DONE__", json.dumps({k: v for k, v in done.items() if k in keys})).replace("__START__", str(start)))


def clicker(real_dir: Path):
    """Colab only: the click page, with the callbacks that serve photos and save each photo's 5 points."""
    from google.colab import output
    from IPython.display import HTML, JSON

    output.register_callback("synthface.photo", lambda key: JSON(photo_for_page(real_dir, key)))

    def save(key: str, points: list | str) -> JSON:
        if points != SKIP and (len(points) != 5 or any(len(p) != 2 for p in points)):
            raise ValueError(f"{key}: expected 5 points")
        save_landmark(real_dir, key, points)
        return JSON({"ok": True})

    output.register_callback("synthface.save", save)
    return HTML(click_page(real_dir))
