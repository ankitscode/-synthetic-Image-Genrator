# CLAUDE.md — SynthFace: synthetic face image generation pipeline

## What this project is

A FastAPI service that generates **synthetic face images** for training a mobile 1:1 face
comparison model (MobileFaceNet + ArcFace). Faces are created with **MPFB2** (MakeHuman plugin
for Blender) and rendered headless with **Blender**. Output: many images per synthetic person
("identity"), aligned to 112×112, with the parameters used to make them.

Target users of the final model are in the **Gulf / Arab region** (very mixed population:
Arab, South Asian, SE Asian, African, Western), so variety in skin tone, face shape,
beards and headwear matters.

## Local setup (this machine)

- **Blender 5.2.2 LTS**, installed as a snap at `/snap/bin/blender` (`BLENDER_PATH`). The snap is
  **pinned** (`snap refresh --hold blender`) — do not upgrade it without asking the user.
- **MPFB2** is installed from the `blender_org` extension repo. Import path:
  `bl_ext.blender_org.mpfb` (verified working headless). MPFB is **not enabled** in the saved
  Blender preferences, so every Blender-side script must call `load_mpfb()` from
  `blender/human.py` first (it enables the add-on for that session only).
- **MakeHuman system assets pack** (CC0, 94 assets, recorded in `assets/LICENSES.csv`) is
  installed in MPFB's user data dir `~/.config/blender/5.2/extensions/.user/blender_org/mpfb/data`
  (outside the repo). Run `blender/inspect_mpfb.py` to list the exact asset names available.
- **Laptop, no NVIDIA GPU**: `RENDER_DEVICE=CPU`, `MAX_CONCURRENT_RENDERS=1`. Cycles on CPU is
  slow — keep test batches small (1–2 identities × 2–4 images) and ask before running anything
  bigger (including the 10 × 10 "first real batch").
- App venv: `venv/` (Python 3.12). Dependencies are pinned in `requirements.txt` (no `pyproject.toml`).
- **Google Colab (GPU)**: `colab/synthface_3d_colab.ipynb` + `colab/setup_colab.sh` install the same
  Blender 5.2.2 / MPFB 2.0.17 / CC0 asset packs / HDRIs (sha256-pinned) on a fresh VM and run
  `render_batch.py` with `"device": "OPTIX"` (or `CUDA`) and `--resume`. Verified: a clean install gives
  identical params to this laptop for the same seed. Bring batches back into `output/<job_id>/`.
  Code is cloned from the public GitHub mirror `https://github.com/ankitscode/-synthetic-Image-Genrator`
  (remote `github`; GitLab `origin` stays). Push both remotes. Open directly:
  `https://colab.research.google.com/github/ankitscode/-synthetic-Image-Genrator/blob/master/colab/synthface_3d_colab.ipynb`
- FastAPI entry point is `main.py` in the project root (not `app/main.py`); the rest of the app
  code lives in `app/`.

## Hard rules (never break these)

1. **License-clean only.** The trained model must be usable commercially with no caveats.
   - Allowed: MPFB2 **core/system assets** (CC0), Blender, assets we make ourselves,
     FLAME 2023 Open **geometry only** (CC-BY-4.0, needs attribution).
   - Not allowed anywhere in this repo or pipeline: FLAME texture model, FLAME Blender add-on,
     older FLAME versions, MetaHuman, Reallusion Character Creator content, DigiFace-1M,
     Digi2Real, DCFace, Vec2Face, HyperFace, any InsightFace model/weights, any
     diffusion/AI-generated faces, any face images scraped from the web.
   - MPFB **community/third-party** asset packs are NOT allowed unless their license is checked
     and recorded.
2. **Every asset gets a row in `assets/LICENSES.csv`** (`path,name,source_url,license,version,added_on,notes`)
   before it is used. If a license is unknown, do not use the asset — ask the user.
3. **No pretrained face detector in this pipeline.** Landmarks come from the 3D model
   (projected vertices), not from a detector.
4. **Deterministic.** Every identity and every image must be reproducible from `(job seed, identity index, image index)`.
   Never call unseeded `random`.
5. **FastAPI never renders in-process.** Rendering always happens in a separate Blender
   subprocess.

## Architecture

```
Client → FastAPI (POST /jobs) → job row in SQLite → runner
                                                     ↓
                    blender --background --python blender/render_batch.py -- --job <job.json>
                                                     ↓
                 output/<job_id>/id_00001/{params.json, raw/*.png, landmarks.json}
                                                     ↓
                 postprocess (normal Python): align → aligned/*.jpg, preview.png
Client ← GET /jobs/{id}, GET /identities/{id}/preview
```

Two separate Python worlds — keep them apart:
- **App side** (`main.py` + `app/`): normal venv, Python 3.11+, FastAPI, Pydantic, OpenCV, NumPy.
- **Blender side** (`blender/`): runs inside Blender's bundled Python. It can import `bpy`,
  MPFB and **only** `shared/` (pure standard-library code). It must not import `app/`,
  FastAPI, Pydantic, OpenCV or anything not shipped with Blender.
- The two sides communicate **only** through JSON files (job spec in, params/landmarks/progress out).

## Directory layout

```
Synthetic Image generator/
  CLAUDE.md
  requirements.txt
  .env / .env.example
  main.py              # FastAPI app + routes (entry point, stays in root)
  app/
    config.py          # pydantic-settings (env vars below)
    schemas.py         # Pydantic request/response models
    jobs.py            # SQLite job store: create/get/update status & progress
    runner.py          # launches Blender subprocess, parses progress, enforces concurrency
    postprocess.py     # 5-point alignment to 112x112, preview sheets
  blender/
    render_batch.py    # entry point run by Blender; reads job.json, renders identities
    human.py           # MPFB calls: create human, skin, eyes, brows, hair, beard
    scene.py           # camera, lights, render settings
    landmarks.py       # 3D → 2D projection of 5 landmarks
  shared/
    sampling.py        # seeded parameter sampling (stdlib only, used by both sides)
    constants.py       # ArcFace template, age mapping, defaults (stdlib only)
  config/
    landmark_vertices.json   # basemesh vertex indices for nose tip + mouth corners
  assets/
    LICENSES.csv
    headwear/ glasses/ beards/   # in-house assets only
  output/              # git-ignored
  tools/
    contact_sheet.py   # Pillow grid for a job: one row per identity, one column per image
    audit_sheet.py     # Pillow grid for blender/debug_*_audit.py output (manifest.json)
  tests/
```

## Commands

```bash
# App
uvicorn main:app --reload
pytest -m "not blender"                 # fast tests, no Blender needed
pytest -m blender                       # integration tests, needs real Blender + MPFB

# Blender directly (debugging one job)
$BLENDER_PATH --background --python blender/render_batch.py -- --job output/<job_id>/job.json

# MPFB check (Blender 5.2.2; MPFB2 installed from the blender_org repo)
$BLENDER_PATH --command extension repo-list
$BLENDER_PATH --background --python-expr "import bl_ext.blender_org.mpfb"
```

Environment variables (`app/config.py`):
`BLENDER_PATH`, `OUTPUT_DIR` (default `./output`), `ASSETS_DIR` (default `./assets`),
`MAX_CONCURRENT_RENDERS` (default 1 per GPU), `RENDER_DEVICE` (`CPU` | `CUDA` | `OPTIX`),
`RENDER_SAMPLES` (default 48). On this machine: `BLENDER_PATH=/snap/bin/blender`,
`RENDER_DEVICE=CPU`, `MAX_CONCURRENT_RENDERS=1`.

## MPFB API notes

Verified signatures (MPFB2 `HumanService`):
- `TargetService.get_default_macro_info_dict()` → default macro dict. **Print it once and use
  its exact keys**; don't guess key names.
- `HumanService.create_human(mask_helpers=True, detailed_helpers=True, extra_vertex_groups=True, feet_on_ground=True, scale=0.1, macro_detail_dict=None)` → basemesh object.
- `HumanService.set_character_skin(mhmat_file, basemesh, bodyproxy=None, skin_type="ENHANCED_SSS", material_instances=True, slot_overrides=None)`.
  Without a skin the character renders grey.
- `HumanService.add_mhclo_asset(mhclo_file, basemesh, asset_type="Clothes", subdiv_levels=1, material_type="MAKESKIN", ...)`
  — used for Eyes, Eyebrows, Eyelashes, Hair, etc. via `asset_type`.
- Find available asset files through `AssetService`; don't hard-code absolute paths.

Gotchas:
- Macro values are 0.0–1.0. Race weights (`african`, `asian`, `caucasian`) are **independent and
  not normalized by MPFB** — normalize them to sum to 1 in `shared/sampling.py`.
- Age is a 0–1 macro, not years. Convert in `shared/constants.py` (MakeHuman convention:
  0.0≈1 yr, 0.1875≈11 yrs, 0.5≈25 yrs, 1.0≈90 yrs, piecewise linear). Verify visually once.
- Extension import path depends on the install repo. Here it is `bl_ext.blender_org.mpfb`
  (verified). Import it once in `blender/human.py`; don't scatter it.
- Fully clear the scene (objects, meshes, materials, images) between identities, or memory grows
  and old assets leak into new renders.

## Rendering rules

- One Blender process renders a **batch of identities** (startup is slow). Never one process per image.
- Engine: **Cycles**, `RENDER_SAMPLES` (32–64) + denoiser, 256×256 output, PNG. Don't use EEVEE
  on headless servers.
- **Move the camera, not the head**, for pose: camera on a sphere around the head, aimed at the
  midpoint between the eyes. Yaw/pitch/roll come from camera placement.
- Lighting: HDRI or sky + one key light; vary colour temperature and intensity per image.
- Expressions (face rig) are **out of scope for v1**.

## Landmarks and alignment

- 5 points, in this order: left eye centre, right eye centre, nose tip, left mouth corner,
  right mouth corner (left/right from the **viewer's** point of view in the image).
- Eye centres: centroid of each eye mesh from the Eyes asset. Nose tip and mouth corners:
  basemesh vertex indices from `config/landmark_vertices.json`. When those indices are first
  chosen or changed, render a debug image with dots drawn on it and ask the user to confirm.
- Project with `bpy_extras.object_utils.world_to_camera_view`, convert to pixel coords
  (origin top-left), and save in `landmarks.json` with a `visible` flag per point.
- Alignment (app side, OpenCV): similarity transform (`cv2.estimateAffinePartial2D`) to the
  standard ArcFace 112×112 template:
  ```python
  ARCFACE_112 = [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
                 [41.5493, 92.3655], [70.7299, 92.2041]]
  ```
  The mobile app must use the same template.
- **Do not bake blur, noise, JPEG artifacts or colour jitter into saved images.** Store clean
  renders; augmentation happens later at training time.

## Parameters (defaults; all overridable per job)

Per identity (fixed for all images of that identity):

| Param | Default range |
|---|---|
| gender | bimodal: 45% 0.0–0.25, 45% 0.75–1.0, 10% 0.25–0.75 |
| age_years | 18–70 (weighted toward 20–50) |
| weight / muscle / proportions | 0.1–0.9 / 0.2–0.8 / 0.2–0.8 |
| race weights | Dirichlet around job mix (default af 0.30 / as 0.35 / ca 0.35), normalized |
| face shape targets | 20–40 categories, strength ±1.0, ≥1 from each of 11 regions (`config/face_targets.json`, list in `output/smoke/face_targets.txt`) |
| skin | light base (asian/caucasian) blended with african base by `tone` (~58% medium, 26% dark, 16% light) + small HSV shift + roughness/specular/clearcoat; once per identity |
| eye colour | by dominant race (african/asian ≥95% brown/darkbrown); `brownlight` unused (renders amber) |
| eyebrows | 12 styles dealt evenly (each once per 12 identities); colour = hair colour × 0.45–1.1 |
| hair | excluded (cover an eye, see `blender/debug_hair_audit.py`): bob01, bob02, braid01, short03 |
| beard (men) | none / stubble / short / full; `beard_prob` 0.5 (not yet: needs a beard asset pack) |

Per image:

| Param | Default range |
|---|---|
| images_per_identity | 10 for tests, 50 for production |
| yaw / pitch / roll | \|yaw\| 60% 0–15°, 30% 15–30°, 10% 30–40°; pitch ±20° (σ 10); roll ±10° (σ 5); re-sampled until both eye centres are visible |
| light colour temp | 3000–6500 K |
| light intensity | 0.3–3× of a 12 W key light (AgX, exposure 0; QA: ≤1% pure-white face pixels) |
| camera focal / distance | 40–55 mm / 45–65 cm; sensor width derived so the face (chin–hairline) fills 55–60% of the frame |
| glasses_prob | 0.25 |
| headwear_prob | 0.35 (only once in-house headwear assets exist) |

## Output format

```
output/<job_id>/
  job.json                      # exact request + resolved defaults + seed
  progress.json                 # {"done": 12, "total": 50, "status": "running"}
  id_00001/
    params.json                 # every sampled identity + per-image parameter
    landmarks.json              # per image: 5 points [x, y] + visible flags
    raw/img_001.png             # 256x256 render
    aligned/img_001.jpg         # 112x112 aligned, JPEG quality 95
    preview.png                 # grid of all aligned images for this identity
```
Identity folders are `id_00001` (5 digits); images are `img_001` (3 digits).

## API

| Method | Path | Purpose |
|---|---|---|
| POST | `/jobs` | start a batch → `{job_id}` (202) |
| GET | `/jobs/{job_id}` | status, progress, error message if failed |
| GET | `/jobs/{job_id}/identities` | list identity ids |
| GET | `/identities/{job_id}/{identity_id}` | params + image list |
| GET | `/identities/{job_id}/{identity_id}/preview` | preview PNG |
| POST | `/preview` | 1 identity × up to 8 images, for testing settings |
| DELETE | `/jobs/{job_id}` | cancel (kill subprocess) and/or delete output |

Limits in `schemas.py`: `num_identities ≤ 1000`, `images_per_identity ≤ 100` per job.
Job states: `queued → running → postprocessing → done | failed | cancelled`.

## Runner behaviour

- Use `asyncio.create_subprocess_exec` (no shell). Enforce `MAX_CONCURRENT_RENDERS` with a semaphore.
- Blender writes progress as one JSON line per finished identity to stdout
  (`{"event": "identity_done", "index": 3}`) and also updates `progress.json`. Ignore other
  stdout lines; keep the last 200 lines of stderr for the error message.
- If one identity fails, log it in `params.json` as failed and continue the batch. If Blender
  exits non-zero, mark the job `failed` with the stderr tail.
- On app startup, mark any `running` jobs as `failed` (interrupted).
- Post-processing runs after the Blender process exits, in a thread pool.

## Code conventions

- Type hints everywhere; Pydantic v2 models for all API input/output.
- `shared/` is stdlib-only and must stay importable from Blender's Python.
- One sampling function per parameter group; each takes an explicit `random.Random` instance.
  Seed per identity = hash of `(job_seed, identity_index)`; per image = `(job_seed, identity_index, image_index)`.
- No magic numbers in code: defaults live in `shared/constants.py` or `schemas.py`.
- Log with `logging`, not `print`, except Blender progress events on stdout.

## Testing

- `tests/test_sampling.py`: same seed → identical params; race weights sum to 1; all values in range.
- `tests/test_postprocess.py`: synthetic landmarks → aligned points land on the ArcFace template (±1 px).
- `tests/test_api.py`: FastAPI `TestClient` with a **fake Blender** (a small script set via
  `BLENDER_PATH` that writes dummy PNGs, landmarks and progress) to test job lifecycle,
  cancellation and failure handling.
- `tests/test_blender_integration.py` (`@pytest.mark.blender`): renders 1 identity × 2 images for real.

## Build order (do these in sequence)

1. `blender/` smoke test: one human, skin, eyes, eyebrows → one PNG. Show it to the user.
2. `shared/sampling.py` + `blender/render_batch.py` for N identities × M images with landmarks.
3. Landmark debug render → user confirms the dots are on the eyes, nose tip and mouth corners.
4. `app/postprocess.py` (alignment + preview sheets).
5. `app/` FastAPI: schemas, job store, runner, endpoints, tests.
6. First real batch: 10 identities × 10 images → user reviews preview sheets.
7. Later: in-house headwear/glasses/beard assets, FLAME 2023 Open geometry (optional), expressions.

## When unsure

- Unknown MPFB key names, asset names or import paths: inspect them inside Blender and print them;
  never guess.
- Any new asset or library with an unclear license: stop and ask the user.
- Visual quality questions (does this look like the same person? are landmarks right?): render a
  preview and ask the user rather than deciding alone.
