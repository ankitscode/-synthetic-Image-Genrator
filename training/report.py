"""HTML evaluation report of a trained run: 1:1 verification and 1:N identification on its held-out people.

    python -m training.report --root <synthface_output> --run <synthface_output>/models/<run> \
        [--cache-dir ...] [--far 0.001 | --threshold 0.45] [--unknown-fraction 0.3]

One threshold for everything (cosine similarity of the 128-d embeddings, "match" = score > threshold): by default the
score that lets 1 in 1,000 different-person pairs through (--far), or a fixed --threshold (e.g. the one the app uses).

- 1:1: every image pair between held-out people (same person / different people), counts at the threshold, a score
  histogram, and example comparisons: random ones, the worst wrong rejections and wrong matches.
- 1:N: each held-out person is enrolled with one image (the most frontal); every other image searches that gallery.
  Closed set: rank-1 / rank-5 and the outcome at the threshold. Open set: --unknown-fraction of the people are not
  enrolled, their images should find no match (wrongly matched share = FPIR). Example searches with the top 5.
- The harder groups of training/hard_eval.py (same sex / tone / hair / look ...).

Writes <run>/report.html: one self-contained file (images embedded), opens in any browser. Synthetic people only:
real-photo accuracy needs a real test set.
"""

from __future__ import annotations

import argparse
import base64
import html
import io
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

from training import hard_eval

EXAMPLES = 12               # example cards per 1:1 section
SEARCH_EXAMPLES = 10        # example searches per 1:N section
PER_PERSON_PAIR = 2         # example cards of the same two people
TOP_K = 5
HIST_BINS = 60
REPORT_SEED = 20261102


# ---------------------------------------------------------------- numbers

def threshold_for_far(impostor: np.ndarray, far: float) -> float:
    """Score that lets about `far` of the different-person pairs through (match = score > threshold)."""
    return float(np.quantile(impostor, 1 - far))


def frontal_image(labels: np.ndarray, image_params: list[dict], person: int) -> int:
    idx = np.flatnonzero(labels == person)
    turn = [abs(image_params[i]["camera"]["yaw"]) + abs(image_params[i]["camera"].get("pitch", 0.0)) for i in idx]
    return int(idx[int(np.argmin(turn))])


def one_to_one(sim: np.ndarray, labels: np.ndarray, threshold: float, rng: np.random.Generator) -> dict:
    i, j = np.triu_indices(len(labels), 1)
    scores, same = sim[i, j], labels[i] == labels[j]
    gen, imp = scores[same], scores[~same]
    counts = {"same_matched": int((gen > threshold).sum()), "same_rejected": int((gen <= threshold).sum()),
              "different_matched": int((imp > threshold).sum()), "different_rejected": int((imp <= threshold).sum())}
    gi, gj, ii, ij = i[same], j[same], i[~same], j[~same]

    def pick(a: np.ndarray, b: np.ndarray, s: np.ndarray, order: np.ndarray) -> list[tuple[int, int, float]]:
        out, seen = [], {}
        for k in order:          # at most PER_PERSON_PAIR cards of the same two people: more variety
            key = (int(labels[a[k]]), int(labels[b[k]]))
            if seen.get(key, 0) < PER_PERSON_PAIR:
                seen[key] = seen.get(key, 0) + 1
                out.append((int(a[k]), int(b[k]), float(s[k])))
                if len(out) == EXAMPLES:
                    break
        return out

    low_gen = np.argsort(gen)
    high_imp = np.argsort(-imp)
    examples = {
        "random same person": pick(gi, gj, gen, rng.permutation(len(gen))),
        "random different people": pick(ii, ij, imp, rng.permutation(len(imp))),
        "same person, lowest scores": pick(gi, gj, gen, low_gen),
        "different people, highest scores": pick(ii, ij, imp, high_imp),
    }
    return {"counts": counts, "genuine": gen, "impostor": imp, "examples": examples}


def one_to_n(sim: np.ndarray, labels: np.ndarray, image_params: list[dict], threshold: float,
             unknown_fraction: float, rng: np.random.Generator) -> dict:
    persons = np.unique(labels)
    enrol = np.array([frontal_image(labels, image_params, p) for p in persons])
    probes = np.setdiff1d(np.arange(len(labels)), enrol)

    def search(gallery_persons: np.ndarray, probe_idx: np.ndarray) -> dict:
        gallery = enrol[np.searchsorted(persons, gallery_persons)]
        s = sim[np.ix_(probe_idx, gallery)]
        order = np.argsort(-s, axis=1)[:, :TOP_K]
        top_persons = gallery_persons[order]
        top_scores = np.take_along_axis(s, order, 1)
        return {"probe": probe_idx, "top_persons": top_persons, "top_scores": top_scores, "gallery": gallery}

    closed = search(persons, probes)
    truth = labels[probes]
    rank1 = closed["top_persons"][:, 0] == truth
    accepted = closed["top_scores"][:, 0] > threshold
    closed_counts = {"searches": int(len(probes)), "gallery": int(len(persons)),
                     "rank1": float(rank1.mean()), "rank5": float((closed["top_persons"] == truth[:, None]).any(1).mean()),
                     "correct_match": int((rank1 & accepted).sum()), "wrong_match": int((~rank1 & accepted).sum()),
                     "no_match": int((~accepted).sum())}

    n_unknown = int(round(unknown_fraction * len(persons)))
    unknown = np.sort(rng.choice(persons, n_unknown, replace=False)) if n_unknown else np.array([], int)
    known = np.setdiff1d(persons, unknown)
    open_probes = probes
    opened = search(known, open_probes)
    t_open = labels[open_probes]
    is_unknown = np.isin(t_open, unknown)
    top1 = opened["top_scores"][:, 0]
    hit = opened["top_persons"][:, 0] == t_open
    unknown_top = top1[is_unknown]
    open_counts = {"gallery": int(len(known)), "unknown_people": int(n_unknown),
                   "known_searches": int((~is_unknown).sum()), "unknown_searches": int(is_unknown.sum()),
                   "known_correct": int((hit & (top1 > threshold) & ~is_unknown).sum()),
                   "known_wrong": int((~hit & (top1 > threshold) & ~is_unknown).sum()),
                   "known_no_match": int(((top1 <= threshold) & ~is_unknown).sum()),
                   "unknown_matched": int((unknown_top > threshold).sum()),
                   "fpir": float((unknown_top > threshold).mean()) if len(unknown_top) else None,
                   "threshold_fpir_1pct": float(np.quantile(unknown_top, 0.99)) if len(unknown_top) else None}

    def rows(result: dict, picks: np.ndarray, truth_known: np.ndarray | None = None) -> list[dict]:
        out = []
        for k in picks[:SEARCH_EXAMPLES]:
            probe = int(result["probe"][k])
            out.append({"probe": probe, "truth": int(labels[probe]),
                        "enrolled": truth_known is None or bool(truth_known[k]),
                        "top": [(int(p), int(enrol[np.searchsorted(persons, p)]), float(s))
                                for p, s in zip(result["top_persons"][k], result["top_scores"][k])]})
        return out

    wrong = np.flatnonzero(~rank1)
    examples = {
        "random searches (everyone enrolled)": rows(closed, rng.permutation(len(probes))),
        "searches where the best match is the wrong person": rows(closed, wrong[np.argsort(-closed["top_scores"][wrong, 0])]),
        "not-enrolled people, highest best match": rows(opened, np.flatnonzero(is_unknown)[np.argsort(-unknown_top)],
                                                        ~is_unknown),
    }
    return {"closed": closed_counts, "open": open_counts, "examples": examples}


# ---------------------------------------------------------------- html

CSS = """
:root{--bg:#f7f7f5;--card:#fff;--ink:#1d1d1b;--muted:#6b6b66;--line:#e2e1dc;--ok:#1f7a4d;--okbg:#e3f3ea;
--bad:#b3261e;--badbg:#fbe5e3;--accent:#2f5d9e;--gen:#1f7a4d;--imp:#b3261e}
@media (prefers-color-scheme:dark){:root{--bg:#161615;--card:#1f1f1d;--ink:#ecebe6;--muted:#a3a29b;--line:#34332f;
--ok:#5cc28d;--okbg:#17321f;--bad:#f08a80;--badbg:#3a1a17;--accent:#86a9e0;--gen:#5cc28d;--imp:#f08a80}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 system-ui,sans-serif}
main{max-width:1100px;margin:0 auto;padding:24px 16px 64px}h1{font-size:24px;margin:0 0 4px}
h2{font-size:19px;margin:36px 0 8px;padding-top:8px;border-top:1px solid var(--line)}h3{font-size:15px;margin:22px 0 8px}
.muted{color:var(--muted)}.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:10px;margin:14px 0}
.tile{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:10px 12px}
.tile b{display:block;font-size:22px;font-variant-numeric:tabular-nums}.tile span{color:var(--muted);font-size:12px}
table{border-collapse:collapse;width:100%;background:var(--card);border:1px solid var(--line);border-radius:8px;
overflow:hidden;font-variant-numeric:tabular-nums}th,td{padding:6px 10px;border-bottom:1px solid var(--line);text-align:right}
th:first-child,td:first-child{text-align:left}th{font-weight:600;color:var(--muted);font-size:12px}
.wrap{overflow-x:auto}.cards{display:grid;grid-template-columns:repeat(auto-fill,minmax(250px,1fr));gap:10px}
.card{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:8px}
.pair{display:flex;gap:6px}.pair figure{margin:0;flex:1;min-width:0}.pair img{width:100%;border-radius:4px;display:block}
figcaption{font-size:11px;color:var(--muted);overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.verdict{display:flex;justify-content:space-between;align-items:center;margin-top:6px;font-size:13px;gap:6px;flex-wrap:wrap}
.badge{border-radius:4px;padding:1px 7px;font-weight:600;font-size:12px}.ok{background:var(--okbg);color:var(--ok)}
.bad{background:var(--badbg);color:var(--bad)}.search{display:grid;grid-template-columns:repeat(6,minmax(0,1fr));gap:6px;
background:var(--card);border:1px solid var(--line);border-radius:8px;padding:8px;margin-bottom:8px}
.search figure{margin:0}.search img{width:100%;border-radius:4px;display:block;border:3px solid transparent}
.search img.hit{border-color:var(--ok)}.search img.miss{border-color:var(--line)}.search .probe img{border-color:var(--accent)}
.note{background:var(--card);border-left:3px solid var(--accent);padding:8px 12px;border-radius:4px;margin:12px 0}
svg text{fill:var(--muted);font-size:11px}
@media (max-width:600px){.search{grid-template-columns:repeat(3,minmax(0,1fr))}}
"""


class Thumbs:
    """Embedded JPEGs, each image encoded once."""

    def __init__(self, images: np.ndarray):
        self.images, self.cache = images, {}

    def __call__(self, idx: int) -> str:
        if idx not in self.cache:
            buf = io.BytesIO()
            Image.fromarray(np.asarray(self.images[idx])).save(buf, "JPEG", quality=88)
            self.cache[idx] = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
        return self.cache[idx]


def pct(v: float | None, digits: int = 2) -> str:
    return "n/a" if v is None else f"{100 * v:.{digits}f}%"


def histogram_svg(gen: np.ndarray, imp: np.ndarray, threshold: float) -> str:
    lo, hi = float(min(gen.min(), imp.min())), 1.0
    edges = np.linspace(lo, hi, HIST_BINS + 1)
    w, h, pad = 900, 220, 30
    x = lambda v: pad + (v - lo) / (hi - lo) * (w - 2 * pad)  # noqa: E731
    parts = []
    for scores, colour, name in ((imp, "var(--imp)", "different people"), (gen, "var(--gen)", "same person")):
        counts = np.histogram(scores, edges)[0] / len(scores)
        top = counts.max() or 1
        pts = " ".join(f"{x(edges[k]):.1f},{h - pad - c / top * (h - 2 * pad):.1f} "
                       f"{x(edges[k + 1]):.1f},{h - pad - c / top * (h - 2 * pad):.1f}" for k, c in enumerate(counts))
        parts.append(f'<polyline points="{x(lo):.1f},{h - pad} {pts} {x(hi):.1f},{h - pad}" fill="{colour}" '
                     f'fill-opacity="0.18" stroke="{colour}" stroke-width="1.5"><title>{name}</title></polyline>')
    ticks = "".join(f'<text x="{x(t):.1f}" y="{h - 10}" text-anchor="middle">{t:.1f}</text>'
                    for t in np.arange(np.ceil(lo * 5) / 5, hi + 1e-9, 0.2))
    tx = x(threshold)
    return (f'<svg viewBox="0 0 {w} {h}" width="100%" role="img" aria-label="score histogram">'
            f'<line x1="{pad}" y1="{h - pad}" x2="{w - pad}" y2="{h - pad}" stroke="var(--line)"/>{"".join(parts)}{ticks}'
            f'<line x1="{tx:.1f}" y1="12" x2="{tx:.1f}" y2="{h - pad}" stroke="var(--accent)" stroke-dasharray="4 3" '
            f'stroke-width="1.5"/><text x="{tx + 4:.1f}" y="22">threshold {threshold:.3f}</text>'
            f'<text x="{pad}" y="14" style="fill:var(--imp)">■ different people</text>'
            f'<text x="{pad + 130}" y="14" style="fill:var(--gen)">■ same person</text></svg>')


def pair_card(thumb: Thumbs, names: list[str], files: list[str], labels: np.ndarray,
              a: int, b: int, score: float, threshold: float) -> str:
    matched, same = score > threshold, labels[a] == labels[b]
    correct = matched == same
    figs = "".join(f'<figure><img src="{thumb(k)}" alt=""><figcaption title="{html.escape(names[labels[k]])}/'
                   f'{html.escape(files[k])}">{html.escape(names[labels[k]])} · {html.escape(files[k])}</figcaption></figure>'
                   for k in (a, b))
    return (f'<div class="card"><div class="pair">{figs}</div><div class="verdict">'
            f'<span>score <b>{score:.3f}</b> {">" if matched else "≤"} {threshold:.3f}</span>'
            f'<span class="badge {"ok" if correct else "bad"}">{"MATCH" if matched else "NO MATCH"}'
            f' · {"same" if same else "different"} · {"correct" if correct else "wrong"}</span></div></div>')


def search_row(thumb: Thumbs, names: list[str], files: list[str], row: dict, threshold: float) -> str:
    best_person, _, best = row["top"][0]
    matched = best > threshold
    if matched:
        correct, text = best_person == row["truth"], f"MATCH → {names[best_person]}"
    else:
        correct, text = not row["enrolled"], "NO MATCH"
    probe = (f'<figure class="probe"><img src="{thumb(row["probe"])}" alt=""><figcaption>search: '
             f'{html.escape(names[row["truth"]])} · {html.escape(files[row["probe"]])}'
             f'{"" if row["enrolled"] else " (not enrolled)"}</figcaption>'
             f'<div class="verdict"><span class="badge {"ok" if correct else "bad"}">{html.escape(text)} · '
             f'{"correct" if correct else "wrong"}</span></div></figure>')
    tops = "".join(f'<figure><img class="{"hit" if p == row["truth"] else "miss"}" src="{thumb(img)}" alt="">'
                   f'<figcaption>{k + 1}. {html.escape(names[p])}</figcaption><figcaption><b>{s:.3f}</b>'
                   f'{" ✓" if s > threshold else ""}</figcaption></figure>'
                   for k, (p, img, s) in enumerate(row["top"]))
    return f'<div class="search">{probe}{tops}</div>'


def render(meta: dict, oo: dict, on: dict, hard: dict, thumb: Thumbs, names: list[str], files: list[str],
           labels: np.ndarray) -> str:
    t, c = meta["threshold"], oo["counts"]
    n_same, n_diff = c["same_matched"] + c["same_rejected"], c["different_matched"] + c["different_rejected"]
    cl, op = on["closed"], on["open"]
    tiles = (f'<div class="tiles"><div class="tile"><b>{t:.3f}</b><span>threshold ({html.escape(meta["threshold_how"])})</span></div>'
             f'<div class="tile"><b>{pct(c["same_matched"] / n_same)}</b><span>1:1 same person matched (TAR)</span></div>'
             f'<div class="tile"><b>{pct(c["different_matched"] / n_diff, 3)}</b><span>1:1 different people matched (FAR)</span></div>'
             f'<div class="tile"><b>{pct(cl["rank1"])}</b><span>1:N rank-1, gallery of {cl["gallery"]}</span></div>'
             f'<div class="tile"><b>{pct(op["fpir"])}</b><span>1:N not-enrolled people matched (FPIR)</span></div></div>')
    head = (f'<h1>Face model evaluation</h1><p class="muted">{html.escape(meta["run"])} · best epoch {meta["epoch"]} · '
            f'{meta["people"]} held-out people · {meta["images"]} images · {meta["created"]}</p>{tiles}'
            '<div class="note">Held-out <b>synthetic</b> people: never seen in training, but rendered like the training '
            'data. Real-photo accuracy needs a real test set. A score is the cosine similarity of the two 128-number '
            'embeddings; <b>match = score &gt; threshold</b>.</div>')

    one = (f'<h2>1:1 verification</h2><p class="muted">Every image pair between the held-out people: is it the same person?</p>'
           f'<div class="wrap"><table><tr><th></th><th>pairs</th><th>MATCH</th><th>NO MATCH</th><th>correct</th></tr>'
           f'<tr><td>same person</td><td>{n_same:,}</td><td>{c["same_matched"]:,}</td><td>{c["same_rejected"]:,}</td>'
           f'<td>{pct(c["same_matched"] / n_same)}</td></tr>'
           f'<tr><td>different people</td><td>{n_diff:,}</td><td>{c["different_matched"]:,}</td>'
           f'<td>{c["different_rejected"]:,}</td><td>{pct(c["different_rejected"] / n_diff, 3)}</td></tr></table></div>'
           f'<h3>Scores</h3>{histogram_svg(oo["genuine"], oo["impostor"], t)}'
           f'<p class="muted">Each curve is scaled to its own peak. Same-person scores left of the line are wrong '
           f'rejections, different-people scores right of it are wrong matches.</p>')
    for title, pairs in oo["examples"].items():
        one += f'<h3>{html.escape(title)}</h3><div class="cards">' + "".join(
            pair_card(thumb, names, files, labels, a, b, s, t) for a, b, s in pairs) + "</div>"

    many = (f'<h2>1:N identification</h2><p class="muted">Each person is enrolled with one image (the most frontal); '
            f'every other image searches the gallery. The best match counts only if its score is above the threshold.</p>'
            f'<h3>Everyone enrolled (closed set)</h3><div class="wrap"><table><tr><th>searches</th><th>gallery</th>'
            f'<th>rank-1</th><th>rank-5</th><th>correct match</th><th>wrong match</th><th>no match</th></tr>'
            f'<tr><td>{cl["searches"]:,}</td><td>{cl["gallery"]}</td><td>{pct(cl["rank1"])}</td><td>{pct(cl["rank5"])}</td>'
            f'<td>{cl["correct_match"]:,} ({pct(cl["correct_match"] / cl["searches"])})</td>'
            f'<td>{cl["wrong_match"]:,} ({pct(cl["wrong_match"] / cl["searches"])})</td>'
            f'<td>{cl["no_match"]:,} ({pct(cl["no_match"] / cl["searches"])})</td></tr></table></div>'
            f'<h3>Some people not enrolled (open set)</h3><p class="muted">{op["unknown_people"]} people left out of the '
            f'gallery ({op["gallery"]} enrolled). Their searches should end in NO MATCH.</p>'
            f'<div class="wrap"><table><tr><th>searches by</th><th>searches</th><th>correct</th><th>wrong person</th>'
            f'<th>no match</th></tr><tr><td>enrolled people</td><td>{op["known_searches"]:,}</td>'
            f'<td>{op["known_correct"]:,}</td><td>{op["known_wrong"]:,}</td><td>{op["known_no_match"]:,}</td></tr>'
            f'<tr><td>not-enrolled people</td><td>{op["unknown_searches"]:,}</td>'
            f'<td>{op["unknown_searches"] - op["unknown_matched"]:,} (no match)</td><td>{op["unknown_matched"]:,} '
            f'(FPIR {pct(op["fpir"])})</td><td>–</td></tr></table></div>'
            + (f'<p class="muted">A gallery search has many chances to match a stranger, so 1:N usually needs a higher '
               f'threshold than 1:1: {op["threshold_fpir_1pct"]:.3f} would let 1% of these strangers match '
               f'(measured on few people: rough).</p>' if op["threshold_fpir_1pct"] is not None else ""))
    for title, rows in on["examples"].items():
        if rows:
            many += f'<h3>{html.escape(title)}</h3><p class="muted">Search image, then the 5 best gallery matches ' \
                    f'(green border = the right person, ✓ = above the threshold).</p>' + "".join(
                        search_row(thumb, names, files, r, t) for r in rows)

    rows = "".join(
        f'<tr><td>{html.escape(name)}</td><td>{r["person_pairs"]:,}</td><td>{r["impostor_pairs"]:,}</td>'
        f'<td>{pct(r.get("far_at_global@0.001"), 3)}</td><td>{pct(r.get("eer"))}</td></tr>'
        for name, r in hard["impostor_groups"].items())
    groups = (f'<h2>Harder groups of different people</h2><p class="muted">Share of different-person pairs matched at '
              f'the FAR 1e-3 threshold of all pairs ({pct(1e-3, 1)} for random pairs). Much more in a group = the model '
              f'leans on what that group shares (sex, skin tone, hair, beard) instead of the face.</p>'
              f'<div class="wrap"><table><tr><th>group</th><th>person pairs</th><th>image pairs</th>'
              f'<th>matched at FAR 1e-3</th><th>EER</th></tr>{rows}</table></div>')
    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" '
            f'content="width=device-width,initial-scale=1"><title>Face model evaluation</title><style>{CSS}</style>'
            f'</head><body><main>{head}{one}{many}{groups}</main></body></html>')


def build(embeddings: np.ndarray, labels: np.ndarray, images: np.ndarray, names: list[str], files: list[str],
          identities: list[dict], image_params: list[dict], shape_rms: np.ndarray | None, meta: dict,
          far: float = 1e-3, threshold: float | None = None, unknown_fraction: float = 0.3) -> tuple[str, dict]:
    """(html, summary dict) from the embeddings of the held-out images."""
    emb = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True).clip(1e-12)
    sim = emb @ emb.T
    rng = np.random.default_rng(REPORT_SEED)
    if threshold is None:
        i, j = np.triu_indices(len(labels), 1)
        threshold = threshold_for_far(sim[i, j][labels[i] != labels[j]], far)
        how = f"FAR {far:g} on these pairs"
    else:
        how = "fixed"
    oo = one_to_one(sim, labels, threshold, rng)
    on = one_to_n(sim, labels, image_params, threshold, unknown_fraction, rng)
    hard = hard_eval.evaluate(embeddings, labels, identities, image_params, shape_rms)
    meta = meta | {"threshold": threshold, "threshold_how": how,
                   "created": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")}
    page = render(meta, oo, on, hard, Thumbs(images), names, files, labels)
    summary = {"threshold": threshold, "threshold_how": how, "one_to_one": oo["counts"],
               "one_to_n_closed": on["closed"], "one_to_n_open": on["open"]}
    return page, summary


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, required=True, help="folder with the chunk folders (synthface_output)")
    p.add_argument("--run", type=Path, required=True, help="trained run folder (best.pt, index.json)")
    p.add_argument("--cache-dir", type=Path, help="default <root>/training_cache")
    p.add_argument("--far", type=float, default=1e-3, help="threshold = score letting this share of strangers match")
    p.add_argument("--threshold", type=float, help="fixed threshold instead of --far")
    p.add_argument("--unknown-fraction", type=float, default=0.3, help="1:N open set: share of people not enrolled")
    p.add_argument("--device", default=None)
    args = p.parse_args()

    holdout, images, labels = hard_eval.load_holdout(args.root, args.run, args.cache_dir or args.root / "training_cache")
    records, image_params = hard_eval.load_params(args.root, holdout)
    embeddings, epoch = hard_eval.embed_best(args.run, images, args.device)
    names = [f"{h.chunk}/{h.identity_id}" for h in holdout]
    files = [Path(rel).stem for h in holdout for rel in h.images]
    meta = {"run": args.run.name, "epoch": epoch, "people": len(holdout), "images": int(len(labels))}
    page, summary = build(embeddings, labels, images, names, files, [r["identity"] for r in records], image_params,
                          hard_eval.face_shapes(args.root, holdout), meta, args.far, args.threshold,
                          args.unknown_fraction)
    (args.run / "report.html").write_text(page, encoding="utf-8")
    (args.run / "report_summary.json").write_text(json.dumps(meta | summary, indent=1) + "\n")
    c, cl, op = summary["one_to_one"], summary["one_to_n_closed"], summary["one_to_n_open"]
    print(f"THRESHOLD {summary['threshold']:.4f} ({summary['threshold_how']})")
    print(f"1:1  same person matched {c['same_matched']}/{c['same_matched'] + c['same_rejected']}  "
          f"different people matched {c['different_matched']}/{c['different_matched'] + c['different_rejected']}")
    print(f"1:N  rank-1 {cl['rank1']:.4f}  rank-5 {cl['rank5']:.4f}  at threshold: correct {cl['correct_match']}  "
          f"wrong {cl['wrong_match']}  no match {cl['no_match']} of {cl['searches']}")
    print(f"1:N open set  not-enrolled matched {op['unknown_matched']}/{op['unknown_searches']} (FPIR {op['fpir']})")
    print(f"wrote {args.run / 'report.html'} ({len(page) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
