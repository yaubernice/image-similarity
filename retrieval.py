"""Post-processing and re-ranking shared by every extractor.

Vectors are L2-normalised before they are indexed. Inner product is
then cosine similarity, and L2 distance is recovered from it.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

PCA_DIM = 256
AQE_K = 5
AQE_ALPHA = 3.0
GEOMETRIC_POOL = 30


def l2_normalize(features: np.ndarray) -> np.ndarray:
    features = np.asarray(features, dtype=np.float32)
    if features.ndim == 1:
        features = features.reshape(1, -1)
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    return features / np.maximum(norms, 1e-10)


def cosine_to_l2(cosine: np.ndarray) -> np.ndarray:
    return np.sqrt(np.maximum(0.0, 2.0 - 2.0 * np.asarray(cosine, dtype=np.float64)))


def similarity_from_l2(distance: float) -> float:
    return 1.0 / (1.0 + float(distance))


def fit_whitening(features: np.ndarray, out_dim: int = PCA_DIM) -> dict:
    """PCA whitening fitted on the database matrix."""
    matrix = np.asarray(features, dtype=np.float64)
    mean = matrix.mean(axis=0)
    centered = matrix - mean
    _, singular, vt = np.linalg.svd(centered, full_matrices=False)
    rank = max(int(np.sum(singular > 1e-6)), 1)
    width = min(out_dim, matrix.shape[1], rank, max(matrix.shape[0] - 1, 1))
    components = vt[:width]
    eigenvalues = (singular[:width] ** 2) / max(matrix.shape[0] - 1, 1)
    scale = 1.0 / np.sqrt(eigenvalues + 1e-6)
    return {
        "mean": mean.astype(np.float32),
        "components": components.astype(np.float32),
        "scale": scale.astype(np.float32),
    }


def apply_whitening(features: np.ndarray, state: dict) -> np.ndarray:
    matrix = np.asarray(features, dtype=np.float32)
    single = matrix.ndim == 1
    if single:
        matrix = matrix.reshape(1, -1)
    projected = (matrix - state["mean"]) @ state["components"].T
    projected = projected * state["scale"]
    normalised = l2_normalize(projected)
    return normalised[0] if single else normalised


def save_whitening(path: Path, state: dict) -> None:
    np.savez(
        path,
        mean=state["mean"],
        components=state["components"],
        scale=state["scale"],
    )


def load_whitening(path: Path) -> dict:
    data = np.load(path)
    return {"mean": data["mean"], "components": data["components"], "scale": data["scale"]}


def expand_query(query: np.ndarray, neighbours: np.ndarray, cosines: np.ndarray, alpha: float = AQE_ALPHA) -> np.ndarray:
    """Average the query with its neighbours, weighted by cosine^alpha."""
    weights = np.power(np.clip(cosines.astype(np.float32), 0.0, None), alpha)
    expanded = query.astype(np.float32) + (weights[:, None] * neighbours).sum(axis=0)
    return l2_normalize(expanded)[0]


def geometric_inliers(query_path: Path, candidate_path: Path) -> int:
    """ORB matches that agree on a RANSAC homography. Zero when there is no geometry."""
    import cv2

    query = cv2.imread(str(query_path), cv2.IMREAD_GRAYSCALE)
    candidate = cv2.imread(str(candidate_path), cv2.IMREAD_GRAYSCALE)
    if query is None or candidate is None:
        return 0
    orb = cv2.ORB_create(nfeatures=800)
    query_kp, query_desc = orb.detectAndCompute(query, None)
    cand_kp, cand_desc = orb.detectAndCompute(candidate, None)
    if query_desc is None or cand_desc is None or len(query_kp) < 8 or len(cand_kp) < 8:
        return 0
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    pairs = matcher.knnMatch(query_desc, cand_desc, k=2)
    good = []
    for pair in pairs:
        if len(pair) < 2:
            continue
        best, second = pair
        if best.distance < 0.75 * second.distance:
            good.append(best)
    if len(good) < 8:
        return 0
    source = np.float32([query_kp[match.queryIdx].pt for match in good])
    dest = np.float32([cand_kp[match.trainIdx].pt for match in good])
    _homography, mask = cv2.findHomography(source, dest, cv2.RANSAC, 5.0)
    if mask is None:
        return 0
    return int(mask.ravel().sum())


def rerank_geometric(
    order: Sequence[int],
    cosines: np.ndarray,
    query_path: Path,
    database_paths: Sequence[Path],
    pool: int = GEOMETRIC_POOL,
) -> List[int]:
    """Move the top of the list around by inlier count, keeping cosine as the tie-break."""
    head = list(order[:pool])
    tail = list(order[pool:])
    scored = []
    for index in head:
        inliers = geometric_inliers(query_path, database_paths[index])
        scored.append((inliers, float(cosines[index]), index))
    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [item[2] for item in scored] + tail
