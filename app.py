"""
Local studio UI for the VGG19 + FAISS image similarity system.

Run:
    python app.py
then open http://127.0.0.1:8787
"""

from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import io
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import quote, unquote

# Quiet TensorFlow before it is imported by the pipeline.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import matplotlib

matplotlib.use("Agg")

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError

import image_similarity as sim

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)

DATABASE_DIR = ROOT / "images" / "database"
QUERY_DIR = ROOT / "images" / "query"
UPLOAD_DIR = ROOT / "images" / "uploads"
INDEX_DIR = ROOT / "index"
STATIC_DIR = ROOT / "static"

for folder in (DATABASE_DIR, QUERY_DIR, UPLOAD_DIR, INDEX_DIR, STATIC_DIR):
    folder.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff"}
MAX_UPLOAD_BYTES = 15 * 1024 * 1024

MODELS = ("vgg19", "dinov2", "dinov2-ft", "clip")
ACTIVE_FILE = INDEX_DIR / "active.txt"

_model_error: Optional[str] = None
_model_loading = False
_loading_name: Optional[str] = None
_inference_lock = threading.Lock()

_indexes: Dict[str, dict] = {}

_state_lock = threading.Lock()
_state: Dict = {
    "building": False,
    "progress": 0.0,
    "message": "Idle",
    "log": [],
    "error": None,
}
_eval_state: Dict = {
    "running": False,
    "message": "Idle",
    "error": None,
    "model": None,
    "variant": None,
}


def _log(message: str) -> None:
    print(message, flush=True)
    with _state_lock:
        _state["log"] = (*_state["log"], message)[-24:]
        _state["message"] = message


def _set_state(**kwargs) -> None:
    with _state_lock:
        _state.update(kwargs)


def active_model() -> str:
    if ACTIVE_FILE.exists():
        name = ACTIVE_FILE.read_text().strip()
        if name in MODELS:
            return name
    return "vgg19"


def set_active_model(name: str) -> None:
    if name not in MODELS:
        raise HTTPException(status_code=400, detail=f"Unknown model '{name}'.")
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    ACTIVE_FILE.write_text(name)


def model_dir(name: str) -> Path:
    return INDEX_DIR / name


def get_extractor(name: str):
    global _model_error, _model_loading, _loading_name
    from extractors import get_extractor as load_extractor, is_loaded, release_others

    if is_loaded(name):
        return load_extractor(name)
    release_others(name)
    _model_loading = True
    _loading_name = name
    _model_error = None
    try:
        _log(f"Loading {name}…")
        extractor = load_extractor(name)
        _log(f"{extractor.label} is ready.")
        return extractor
    except Exception as exc:  # noqa: BLE001
        _model_error = str(exc)
        _log(f"{name} failed to load: {exc}")
        raise
    finally:
        _model_loading = False
        _loading_name = None


_ft_worker: Optional[subprocess.Popen] = None
_ft_worker_lock = threading.Lock()


def _fine_tuned_vectors(paths: List[Path], tta: bool, batch_size: int) -> np.ndarray:
    """Embed with dinov2-ft in a child process.

    The checkpoint cannot be loaded in a process that has already imported
    FAISS; that combination segfaults. The child never imports FAISS.
    """
    global _ft_worker
    with _ft_worker_lock:
        if _ft_worker is None or _ft_worker.poll() is not None:
            _ft_worker = subprocess.Popen(
                [sys.executable, str(ROOT / "extractors.py"), "--worker", "dinov2-ft"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                text=True,
                cwd=str(ROOT),
            )
        assert _ft_worker.stdin is not None and _ft_worker.stdout is not None
        with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as handle:
            out_path = handle.name
        request = json.dumps({
            "paths": [str(path) for path in paths],
            "tta": tta,
            "batch": batch_size,
            "out": out_path,
        })
        _ft_worker.stdin.write(request + "\n")
        _ft_worker.stdin.flush()
        reply_line = _ft_worker.stdout.readline()
        if not reply_line:
            raise RuntimeError("The fine-tuned embedder stopped before returning features.")
        reply = json.loads(reply_line)
        if not reply.get("ok"):
            raise RuntimeError(reply.get("error", "Fine-tuned embedding failed."))
        features = np.load(out_path)
        os.unlink(out_path)
        return features


def _warm_model() -> None:
    try:
        get_extractor(active_model())
    except Exception:
        return


def get_index(model: Optional[str] = None):
    """Load one model's FAISS index, features, and whitening."""
    name = model or active_model()
    directory = model_dir(name)
    index_path = directory / "faiss.index"
    if not index_path.exists():
        _indexes.pop(name, None)
        return None
    mtime = index_path.stat().st_mtime
    cached = _indexes.get(name)
    if cached and cached["mtime"] == mtime:
        return cached
    index, paths, features = sim.load_index(directory)
    meta_path = directory / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    whitening = None
    pca_path = directory / "pca.npz"
    if meta.get("pca") and pca_path.exists():
        from retrieval import load_whitening
        whitening = load_whitening(pca_path)
    _indexes[name] = {
        "mtime": mtime,
        "model": name,
        "index": index,
        "paths": paths,
        "features": features,
        "meta": meta,
        "whitening": whitening,
        "space": None,
    }
    return _indexes[name]


def _database_images() -> List[Path]:
    if not DATABASE_DIR.exists():
        return []
    return sorted(
        p for p in DATABASE_DIR.rglob("*")
        if p.is_file() and p.suffix.lower() in sim.IMAGE_EXTENSIONS
    )


def _rel_to(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _media_url(root: Path, path: Path, mount: str) -> str:
    return f"{mount}/{quote(_rel_to(root, path))}"


def _safe_file(root: Path, rel: str) -> Path:
    rel = unquote(rel).replace("\\", "/").lstrip("/")
    if not rel or any(part == ".." for part in Path(rel).parts):
        raise HTTPException(status_code=404, detail="Image not found")
    path = (root / rel).resolve()
    root_resolved = root.resolve()
    if path != root_resolved and root_resolved not in path.parents:
        raise HTTPException(status_code=404, detail="Image not found")
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Image not found")
    return path


def _read_meta(model: Optional[str] = None) -> dict:
    cache = get_index(model)
    if cache:
        return cache["meta"]
    meta_path = model_dir(model or active_model()) / "meta.json"
    if not meta_path.exists():
        return {}
    try:
        return json.loads(meta_path.read_text())
    except Exception:
        return {}


def _indexed_path_set(cache) -> set:
    if not cache:
        return set()
    found = set()
    for path in cache["paths"]:
        try:
            resolved = Path(path).resolve()
        except OSError:
            continue
        if resolved.exists():
            found.add(resolved)
    return found


def _model_summary(name: str, images: set) -> dict:
    from extractors import is_loaded

    cache = get_index(name)
    meta = cache["meta"] if cache else {}
    index_path = model_dir(name) / "faiss.index"
    indexed_paths = _indexed_path_set(cache)
    new_count = len(images - indexed_paths)
    missing_count = len(indexed_paths - images)
    mtime = index_path.stat().st_mtime if index_path.exists() else None
    return {
        "key": name,
        "label": meta.get("model", name),
        "indexed": cache is not None,
        "num_vectors": int(cache["index"].ntotal) if cache else 0,
        "feature_dim": int(cache["features"].shape[1]) if cache else meta.get("feature_dim"),
        "built_at": datetime.fromtimestamp(mtime).strftime("%d %b %Y, %H:%M") if mtime else None,
        "stale": bool(cache and (new_count or missing_count)),
        "new_count": new_count,
        "missing_count": missing_count,
        "loaded": is_loaded(name),
        "tta": bool(meta.get("tta")),
        "pca": bool(meta.get("pca")),
    }


def status_payload() -> dict:
    images = _database_images()
    image_set = {path.resolve() for path in images}
    selected = active_model()
    models = {name: _model_summary(name, image_set) for name in MODELS}
    current = models[selected]
    with _state_lock:
        build = {
            "building": _state["building"],
            "progress": _state["progress"],
            "message": _state["message"],
            "log": list(_state["log"]),
            "error": _state["error"],
        }
        evaluation = dict(_eval_state)
    return {
        "database_count": len(images),
        "query_count": len(list(QUERY_DIR.glob("*"))) if QUERY_DIR.exists() else 0,
        "active_model": selected,
        "models": models,
        "indexed": current["indexed"],
        "num_vectors": current["num_vectors"],
        "feature_dim": current["feature_dim"],
        "model": current["label"],
        "built_at": current["built_at"],
        "stale": current["stale"],
        "new_count": current["new_count"],
        "missing_count": current["missing_count"],
        "model_loaded": current["loaded"],
        "model_loading": _model_loading and _loading_name == selected,
        "model_error": _model_error,
        "evaluation": evaluation,
        **build,
    }


def _fingerprint(vector: np.ndarray, bins: int = 48) -> List[float]:
    parts = np.array_split(np.asarray(vector, dtype=np.float64).reshape(-1), bins)
    return [0.0 if part.size == 0 else round(float(part.mean()), 5) for part in parts]


def _feature_space(cache) -> dict:
    """Project the index into 2D with PCA so the UI can draw the feature map."""
    if cache.get("space") and cache["space"]["mtime"] == cache["mtime"]:
        return cache["space"]

    features = cache["features"].astype(np.float64)
    mean = features.mean(axis=0)
    centered = features - mean
    _, singular, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[:2]
    coords = centered @ components.T
    mins = coords.min(axis=0)
    span = np.maximum(coords.max(axis=0) - mins, 1e-9)
    variance = singular ** 2
    total = float(variance.sum()) or 1.0
    explained = [float(variance[i] / total) if i < len(variance) else 0.0 for i in range(2)]

    cache["space"] = {
        "mtime": cache["mtime"],
        "mean": mean,
        "components": components,
        "mins": mins,
        "span": span,
        "coords": coords,
        "explained": explained,
    }
    return cache["space"]


def _to_map_point(coord: np.ndarray, space: dict) -> dict:
    normed = (coord - space["mins"]) / space["span"]
    padded = 0.14 + normed * 0.72
    return {"x": float(padded[0]), "y": float(padded[1])}


def _project_query(vector: np.ndarray, space: dict) -> dict:
    coord = (vector.astype(np.float64) - space["mean"]) @ space["components"].T
    return _to_map_point(coord, space)


def space_payload() -> dict:
    cache = get_index()
    if cache is None:
        return {"explained": [], "points": []}
    space = _feature_space(cache)
    points = []
    for path, coord in zip(cache["paths"], space["coords"]):
        resolved = Path(path)
        if not resolved.exists():
            continue
        try:
            url = _media_url(DATABASE_DIR, resolved, "/media/database")
            rel = _rel_to(DATABASE_DIR, resolved)
        except ValueError:
            continue
        point = _to_map_point(coord, space)
        points.append({
            "name": resolved.name,
            "path": rel,
            "url": url,
            "x": point["x"],
            "y": point["y"],
        })
    return {
        "explained": space["explained"],
        "points": points,
    }


def _unique_destination(folder: Path, filename: str) -> Path:
    safe = Path(filename).name
    stem = Path(safe).stem or "image"
    suffix = Path(safe).suffix.lower()
    candidate = folder / f"{stem}{suffix}"
    counter = 2
    while candidate.exists():
        candidate = folder / f"{stem}-{counter}{suffix}"
        counter += 1
    return candidate


def _validate_image(data: bytes, filename: str) -> None:
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"{filename} is not a supported image. Use JPEG, PNG, WebP, GIF, BMP, or TIFF.",
        )
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=400, detail=f"{filename} is larger than 15 MB.")
    if not data:
        raise HTTPException(status_code=400, detail=f"{filename} is empty.")
    try:
        with Image.open(io.BytesIO(data)) as image:
            image.load()
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(status_code=400, detail=f"{filename} could not be read as an image.") from exc


def _build_index(model: str) -> None:
    from retrieval import apply_whitening, fit_whitening, save_whitening

    try:
        _set_state(building=True, progress=0.02, error=None, message="Collecting photographs")
        paths = [path.resolve() for path in _database_images()]
        if not paths:
            raise RuntimeError("The library is empty. Add photographs before building an index.")
        _log(f"Found {len(paths)} photographs.")
        _set_state(progress=0.06, message=f"Preparing {model}")
        with _inference_lock:
            label = "DINOv2 fine-tuned (triplet)"
            if model == "dinov2-ft":
                _log("Embedding with the fine-tuned model.")
                features = _fine_tuned_vectors(paths, tta=True, batch_size=8)
            else:
                extractor = get_extractor(model)
                label = extractor.label

                def progress(done: int, total: int) -> None:
                    _set_state(
                        progress=0.08 + 0.72 * (done / max(total, 1)),
                        message=f"Extracting features · {done} of {total}",
                    )

                features = extractor.embed(
                    paths,
                    batch_size=8 if model.startswith("dinov2") else 16,
                    tta=True,
                    progress=progress,
                )
            _set_state(progress=0.84, message="Whitening the descriptors")
            directory = model_dir(model)
            directory.mkdir(parents=True, exist_ok=True)
            state = fit_whitening(features)
            features = apply_whitening(features, state)
            save_whitening(directory / "pca.npz", state)
            _set_state(progress=0.94, message="Building the FAISS index")
            index = sim.build_faiss_index(features, metric="ip")
        sim.save_index(
            index,
            paths,
            features,
            directory,
            model=label,
            metric="ip",
            extra={
                "model_key": model,
                "tta": True,
                "pca": True,
                "pooling": "gem" if model == "vgg19" else "triplet" if model == "dinov2-ft" else "native",
                "built_at": datetime.now().isoformat(timespec="seconds"),
            },
        )
        set_active_model(model)
        _indexes.pop(model, None)
        get_index(model)
        _log(f"Index saved. {index.ntotal} vectors, dimension {features.shape[1]}.")
        _set_state(progress=1.0, message=f"Index ready · {index.ntotal} vectors", error=None)
    except Exception as exc:  # noqa: BLE001
        _set_state(error=str(exc), message="Index build failed")
        _log(f"Index build failed: {exc}")
    finally:
        _set_state(building=False)


def _embed_query_vector(cache: dict, path: Optional[Path] = None, text: Optional[str] = None) -> np.ndarray:
    from retrieval import apply_whitening

    meta = cache["meta"]
    name = meta.get("model_key", cache["model"])
    tta = bool(meta.get("tta", True))
    with _inference_lock:
        if text is not None:
            if name != "clip":
                raise HTTPException(status_code=400, detail="Text search uses the CLIP model.")
            vector = get_extractor(name).embed_text(text)
        elif name == "dinov2-ft":
            vector = _fine_tuned_vectors([path], tta=tta, batch_size=1)
        else:
            vector = get_extractor(name).embed([path], tta=tta)
    if cache["whitening"] is not None:
        vector = apply_whitening(vector, cache["whitening"])
    return np.asarray(vector, dtype=np.float32).reshape(-1)


def _rank_vector(cache: dict, vector: np.ndarray, top_k: int, aqe: bool, geometric: bool, query_path: Optional[Path]) -> list:
    from retrieval import AQE_K, cosine_to_l2, expand_query, rerank_geometric, similarity_from_l2

    metric = cache["meta"].get("metric", "l2")
    total = int(cache["index"].ntotal)
    pool = min(total, 30 if geometric else max(top_k, AQE_K if aqe else top_k))
    scores, indices = cache["index"].search(np.ascontiguousarray(vector.reshape(1, -1)), pool)
    order = [int(item) for item in indices[0] if item >= 0]
    raw_scores = {index: float(scores[0, rank]) for rank, index in enumerate(order)}

    def cosine_of(index: int) -> float:
        if metric == "ip":
            return raw_scores.get(index, float(np.dot(cache["features"][index], vector)))
        return float(np.dot(cache["features"][index], vector))

    if aqe and order:
        top = order[: min(AQE_K, len(order))]
        cosines = np.array([cosine_of(index) for index in top], dtype=np.float32)
        vector = expand_query(vector, cache["features"][top], cosines)
        scores, indices = cache["index"].search(np.ascontiguousarray(vector.reshape(1, -1)), pool)
        order = [int(item) for item in indices[0] if item >= 0]
        raw_scores = {index: float(scores[0, rank]) for rank, index in enumerate(order)}

    cosines = np.array([cosine_of(index) for index in range(total)], dtype=np.float32)
    if geometric and query_path is not None:
        order = rerank_geometric(order, cosines, query_path, [Path(path) for path in cache["paths"]])

    space = _feature_space(cache)
    results = []
    for index in order[:top_k]:
        path = Path(cache["paths"][index])
        if not path.exists():
            continue
        try:
            url = _media_url(DATABASE_DIR, path, "/media/database")
            rel = _rel_to(DATABASE_DIR, path)
        except ValueError:
            continue
        cosine = float(cosines[index])
        distance = float(cosine_to_l2(cosine)) if metric == "ip" else float(raw_scores.get(index, 0.0))
        if metric != "ip":
            cosine = float(np.dot(cache["features"][index], vector))
        point = _to_map_point(space["coords"][index], space)
        results.append({
            "rank": len(results) + 1,
            "name": path.name,
            "path": rel,
            "url": url,
            "distance": round(distance, 4),
            "cosine": round(cosine, 4),
            "similarity": round(similarity_from_l2(distance), 4),
            "fingerprint": _fingerprint(cache["features"][index]),
            "x": point["x"],
            "y": point["y"],
        })
    return results, vector


def _search_sync(
    data: bytes,
    filename: str,
    top_k: int,
    model: Optional[str] = None,
    aqe: bool = False,
    geometric: bool = False,
    destination: Optional[Path] = None,
) -> dict:
    name = model or active_model()
    cache = get_index(name)
    if cache is None:
        raise HTTPException(status_code=409, detail=f"Build a {name} index in the Library before searching.")
    if destination is None:
        _validate_image(data, filename)
        destination = _unique_destination(UPLOAD_DIR, f"query-{uuid.uuid4().hex[:8]}-{Path(filename).name}")
        destination.write_bytes(data)

    started = time.perf_counter()
    try:
        query_vector = _embed_query_vector(cache, path=destination)
        results, query_vector = _rank_vector(cache, query_vector, top_k, aqe, geometric, destination)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Could not read that image: {exc}") from exc

    space = _feature_space(cache)
    elapsed_ms = round((time.perf_counter() - started) * 1000)
    return {
        "model": name,
        "query_url": _media_url(UPLOAD_DIR, destination, "/media/uploads"),
        "query_name": Path(filename).name,
        "elapsed_ms": elapsed_ms,
        "searched": int(cache["index"].ntotal),
        "vector_norm": round(float(np.linalg.norm(query_vector)), 4),
        "fingerprint": _fingerprint(query_vector),
        "query_point": _project_query(query_vector, space),
        "aqe": aqe,
        "geometric": geometric,
        "results": results,
    }


def _search_text_sync(text: str, top_k: int, aqe: bool) -> dict:
    cache = get_index("clip")
    if cache is None:
        raise HTTPException(status_code=409, detail="Build a CLIP index before searching with text.")
    started = time.perf_counter()
    query_vector = _embed_query_vector(cache, text=text.strip())
    results, query_vector = _rank_vector(cache, query_vector, top_k, aqe, False, None)
    space = _feature_space(cache)
    return {
        "model": "clip",
        "query_text": text.strip(),
        "elapsed_ms": round((time.perf_counter() - started) * 1000),
        "vector_norm": round(float(np.linalg.norm(query_vector)), 4),
        "searched": int(cache["index"].ntotal),
        "fingerprint": _fingerprint(query_vector),
        "query_point": _project_query(query_vector, space),
        "aqe": aqe,
        "geometric": False,
        "results": results,
    }


def _saved_evaluations() -> list:
    directory = ROOT / "results"
    if not directory.exists():
        return []
    rows = []
    for path in sorted(directory.glob("eval_*.json")):
        try:
            rows.append(json.loads(path.read_text()))
        except json.JSONDecodeError:
            continue
    return rows


def _run_evaluation(model: str, variant: str) -> None:
    global _eval_state
    try:
        from evaluate import DATA, evaluate_baseline, evaluate_variant, load_ground_truth, write_summary

        ground_path = DATA / "ground_truth.json"
        if not ground_path.exists():
            raise FileNotFoundError("Run scripts/prepare_caltech101.py before evaluating.")
        ground = load_ground_truth(ground_path)
        _eval_state["message"] = f"Scoring {model} / {variant}"
        if model == "dinov2-ft":
            completed = subprocess.run(
                [sys.executable, str(ROOT / "evaluate.py"), "--model", model, "--variant", variant, "--batch-size", "8"],
                cwd=str(ROOT),
                capture_output=True,
                text=True,
            )
            if completed.returncode != 0:
                detail = (completed.stderr or completed.stdout or "evaluation failed").strip()
                raise RuntimeError(detail[-800:])
            summary = json.loads((ROOT / "results" / f"eval_{model}_{variant}.json").read_text())
        else:
            with _inference_lock:
                if variant == "baseline":
                    summary = evaluate_baseline(ground)
                else:
                    summary = evaluate_variant(ground, model, variant, 16, DATA)
            write_summary(summary)
        _eval_state["message"] = f"mAP {summary['mAP']:.3f}"
        _eval_state["error"] = None
    except Exception as exc:  # noqa: BLE001
        _eval_state["error"] = str(exc)
        _eval_state["message"] = "Evaluation failed"
    finally:
        _eval_state["running"] = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    threading.Thread(target=_warm_model, name="vgg19-warmup", daemon=True).start()
    yield


app = FastAPI(title="Image Similarity", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def home():
    return FileResponse(STATIC_DIR / "index.html", headers={"Cache-Control": "no-cache"})


@app.get("/api/status")
def api_status():
    return status_payload()


@app.get("/api/library")
def api_library():
    cache = get_index()
    indexed = _indexed_path_set(cache)
    items = []
    for path in _database_images():
        items.append({
            "name": path.name,
            "path": _rel_to(DATABASE_DIR, path),
            "url": _media_url(DATABASE_DIR, path, "/media/database"),
            "indexed": path.resolve() in indexed,
        })
    return {"images": items}


@app.get("/api/samples")
def api_samples():
    images = sorted(
        p for p in QUERY_DIR.rglob("*")
        if p.is_file() and p.suffix.lower() in sim.IMAGE_EXTENSIONS
    )
    return {
        "images": [
            {
                "name": path.name,
                "url": _media_url(QUERY_DIR, path, "/media/query"),
            }
            for path in images
        ]
    }


@app.get("/api/space")
def api_space():
    return space_payload()


@app.post("/api/library")
async def api_upload(files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="Choose at least one image.")
    saved = []
    for upload in files:
        data = await upload.read()
        filename = upload.filename or "image.jpg"
        _validate_image(data, filename)
        destination = _unique_destination(DATABASE_DIR, filename)
        destination.write_bytes(data)
        saved.append(destination.name)
    return {"saved": saved, "count": len(saved)}


@app.delete("/api/library/{file_path:path}")
def api_delete(file_path: str):
    path = _safe_file(DATABASE_DIR, file_path)
    path.unlink()
    return {"ok": True, "removed": path.name}


def _flag(value: str) -> bool:
    return str(value).lower() in {"1", "true", "yes", "on"}


@app.post("/api/model")
def api_model(model: str = Form(...)):
    set_active_model(model)
    return {"active_model": active_model()}


@app.post("/api/index/build")
def api_build(model: str = Form("vgg19")):
    if model not in MODELS:
        raise HTTPException(status_code=400, detail=f"Unknown model '{model}'.")
    with _state_lock:
        if _state["building"]:
            raise HTTPException(status_code=409, detail="An index build is already running.")
        _state["building"] = True
        _state["progress"] = 0.01
        _state["error"] = None
        _state["log"] = []
        _state["message"] = "Starting"
    threading.Thread(target=_build_index, args=(model,), name="faiss-build", daemon=True).start()
    return {"ok": True, "model": model}


@app.post("/api/search")
async def api_search(
    file: UploadFile = File(...),
    top_k: int = Form(8),
    model: str = Form(""),
    aqe: str = Form("false"),
    geometric: str = Form("false"),
):
    data = await file.read()
    filename = file.filename or "query.jpg"
    import asyncio
    chosen = model if model in MODELS else None
    return await asyncio.to_thread(
        _search_sync, data, filename, int(top_k), chosen, _flag(aqe), _flag(geometric), None
    )


@app.post("/api/search/text")
async def api_search_text(text: str = Form(...), top_k: int = Form(8), aqe: str = Form("false")):
    if not text.strip():
        raise HTTPException(status_code=400, detail="Enter a text query.")
    import asyncio
    return await asyncio.to_thread(_search_text_sync, text, int(top_k), _flag(aqe))


@app.post("/api/compare")
async def api_compare(file: UploadFile = File(...), top_k: int = Form(6)):
    data = await file.read()
    filename = file.filename or "query.jpg"
    _validate_image(data, filename)
    destination = _unique_destination(UPLOAD_DIR, f"query-{uuid.uuid4().hex[:8]}-{Path(filename).name}")
    destination.write_bytes(data)
    import asyncio

    def run():
        columns = []
        for name in MODELS:
            if get_index(name) is None:
                columns.append({"model": name, "indexed": False, "results": []})
                continue
            payload = _search_sync(data, filename, int(top_k), name, False, False, destination)
            columns.append({"model": name, "indexed": True, **payload})
        return {"query_url": _media_url(UPLOAD_DIR, destination, "/media/uploads"), "columns": columns}

    return await asyncio.to_thread(run)


@app.get("/api/evaluate")
def api_evaluate_results():
    return {"state": _eval_state, "results": _saved_evaluations(), "ready": (ROOT / "data" / "caltech101" / "ground_truth.json").exists()}


@app.post("/api/evaluate")
def api_evaluate(model: str = Form(...), variant: str = Form("plain")):
    if model not in MODELS:
        raise HTTPException(status_code=400, detail=f"Unknown model '{model}'.")
    if variant not in {"baseline", "plain", "tta_pca", "aqe", "geom"}:
        raise HTTPException(status_code=400, detail=f"Unknown variant '{variant}'.")
    if variant == "baseline" and model != "vgg19":
        raise HTTPException(status_code=400, detail="The baseline is the original VGG19 global-average-pool index.")
    if _eval_state["running"] or _state["building"]:
        raise HTTPException(status_code=409, detail="An evaluation or index build is already running.")
    _eval_state.update({"running": True, "message": "Starting", "error": None, "model": model, "variant": variant})
    threading.Thread(target=_run_evaluation, args=(model, variant), name="evaluate", daemon=True).start()
    return {"ok": True}


@app.get("/media/database/{file_path:path}")
def media_database(file_path: str):
    return FileResponse(_safe_file(DATABASE_DIR, file_path))


@app.get("/media/query/{file_path:path}")
def media_query(file_path: str):
    return FileResponse(_safe_file(QUERY_DIR, file_path))


@app.get("/media/uploads/{file_path:path}")
def media_uploads(file_path: str):
    return FileResponse(_safe_file(UPLOAD_DIR, file_path))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="127.0.0.1", port=8787, reload=False)
