"""1:1 verification metrics on held-out people (numpy only).

Pairs: every same-person pair (genuine) and the same number of random different-person pairs (impostor),
drawn with a fixed seed. Score = cosine similarity of L2-normalised embeddings.
"""

from __future__ import annotations

import numpy as np

FARS = (1e-2, 1e-3, 1e-4)


def make_pairs(labels: np.ndarray, seed: int = 0, max_genuine: int = 200_000) -> tuple[np.ndarray, np.ndarray]:
    """(pairs (P, 2) of image indices, is_same (P,) bool)."""
    rng = np.random.default_rng(seed)
    genuine = []
    for label in np.unique(labels):
        idx = np.flatnonzero(labels == label)
        a, b = np.triu_indices(len(idx), 1)
        genuine.append(np.stack([idx[a], idx[b]], 1))
    genuine = np.concatenate(genuine) if genuine else np.zeros((0, 2), np.int64)
    if len(genuine) > max_genuine:
        genuine = genuine[rng.choice(len(genuine), max_genuine, replace=False)]
    impostor = []
    need = len(genuine)
    while need > 0:
        cand = rng.integers(0, len(labels), (need * 2, 2))
        cand = cand[labels[cand[:, 0]] != labels[cand[:, 1]]][:need]
        impostor.append(cand)
        need -= len(cand)
    impostor = np.concatenate(impostor) if impostor else np.zeros((0, 2), np.int64)
    pairs = np.concatenate([genuine, impostor])
    same = np.concatenate([np.ones(len(genuine), bool), np.zeros(len(impostor), bool)])
    return pairs, same


def verification(embeddings: np.ndarray, labels: np.ndarray, seed: int = 0) -> dict:
    """AUC, EER, TAR at FAR 1e-2 / 1e-3 / 1e-4, best accuracy and its threshold."""
    emb = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True).clip(1e-12)
    pairs, same = make_pairs(labels, seed)
    scores = (emb[pairs[:, 0]] * emb[pairs[:, 1]]).sum(1)
    order = np.argsort(-scores, kind="stable")
    s, y = scores[order], same[order]
    tp = np.cumsum(y) / max(y.sum(), 1)            # TAR when accepting the top k
    fp = np.cumsum(~y) / max((~y).sum(), 1)        # FAR when accepting the top k
    trapezoid = getattr(np, "trapezoid", None) or np.trapz  # numpy < 2 names it trapz
    auc = float(trapezoid(np.concatenate([[0], tp]), np.concatenate([[0], fp])))
    eer_i = int(np.argmin(np.abs((1 - tp) - fp)))
    acc = (np.cumsum(y) + ((~y).sum() - np.cumsum(~y))) / len(y)
    best = int(np.argmax(acc))
    out = {"pairs_genuine": int(same.sum()), "pairs_impostor": int((~same).sum()), "auc": auc,
           "eer": float(((1 - tp[eer_i]) + fp[eer_i]) / 2), "best_accuracy": float(acc[best]),
           "best_threshold": float(s[best])}
    for far in FARS:
        ok = np.flatnonzero(fp <= far)
        out[f"tar@far={far:g}"] = float(tp[ok[-1]]) if len(ok) else 0.0
        out[f"threshold@far={far:g}"] = float(s[ok[-1]]) if len(ok) else 1.0
    out["genuine_score_mean"] = float(scores[same].mean())
    out["impostor_score_mean"] = float(scores[~same].mean())
    return out
