"""Download Caltech-101 and build a 12-class retrieval subset.

Writes:
  data/caltech101/database/<class>/   30 images
  data/caltech101/queries/<class>/     3 held-out images
  data/caltech101/ground_truth.json

A query is relevant to every database image of the same class.
The query itself is not copied into the database.
"""

from __future__ import annotations

import json
import random
import shutil
import tarfile
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "caltech101"
ARCHIVE = ROOT / "data" / "caltech-101.zip"
URL = "https://data.caltech.edu/records/mzrjq-6wc02/files/caltech-101.zip?download=1"

PREFERRED = [
    "airplanes",
    "Motorbikes",
    "Faces_easy",
    "watch",
    "Leopards",
    "bonsai",
    "car_side",
    "butterfly",
    "chandelier",
    "helicopter",
    "kangaroo",
    "sunflower",
]
DATABASE_PER_CLASS = 30
QUERIES_PER_CLASS = 3
NEEDED = DATABASE_PER_CLASS + QUERIES_PER_CLASS
SEED = 20
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def download() -> None:
    ARCHIVE.parent.mkdir(parents=True, exist_ok=True)
    if ARCHIVE.exists() and zipfile.is_zipfile(ARCHIVE):
        print(f"Archive already present: {ARCHIVE}")
        return
    if ARCHIVE.exists() and not zipfile.is_zipfile(ARCHIVE):
        ARCHIVE.unlink()
    print(f"Downloading Caltech-101 → {ARCHIVE}")
    request = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=120) as response, ARCHIVE.open("wb") as handle:
        shutil.copyfileobj(response, handle)
    if not zipfile.is_zipfile(ARCHIVE):
        raise RuntimeError(f"Download did not produce a zip file ({ARCHIVE.stat().st_size} bytes)")
    print(f"Downloaded {ARCHIVE.stat().st_size} bytes")


def category_root(extracted: Path) -> Path:
    candidates = list(extracted.rglob("101_ObjectCategories"))
    if not candidates:
        raise FileNotFoundError("101_ObjectCategories was not inside the archive")
    return candidates[0]


def choose_classes(root: Path) -> list[tuple[str, list[Path]]]:
    folders = {}
    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        if folder.name.lower() in {"background_google", "background"}:
            continue
        images = sorted(
            p for p in folder.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
        )
        if len(images) >= NEEDED:
            folders[folder.name.lower()] = (folder.name, images)

    chosen = []
    used = set()
    for name in PREFERRED:
        key = name.lower()
        if key in folders and key not in used:
            chosen.append(folders[key])
            used.add(key)
    if len(chosen) < 12:
        rest = sorted(folders.values(), key=lambda item: len(item[1]), reverse=True)
        for name, images in rest:
            if name.lower() in used:
                continue
            chosen.append((name, images))
            used.add(name.lower())
            if len(chosen) == 12:
                break
    if len(chosen) < 12:
        raise RuntimeError(f"Only found {len(chosen)} classes with at least {NEEDED} images")
    return chosen[:12]


def ensure_categories(extracted: Path) -> Path:
    """The official zip wraps 101_ObjectCategories.tar.gz rather than the folders."""
    if category_root_exists(extracted):
        return category_root(extracted)
    extracted.mkdir(parents=True, exist_ok=True)
    if ARCHIVE.exists() and not any(extracted.rglob("101_ObjectCategories.tar.gz")):
        print(f"Extracting {ARCHIVE}")
        with zipfile.ZipFile(ARCHIVE) as archive:
            archive.extractall(extracted)
    archives = list(extracted.rglob("101_ObjectCategories.tar.gz"))
    if not archives:
        raise FileNotFoundError("101_ObjectCategories was not inside the archive")
    print(f"Extracting {archives[0].name}")
    with tarfile.open(archives[0]) as archive:
        archive.extractall(extracted, filter="data")
    return category_root(extracted)


def prepare() -> None:
    download()
    extracted = ROOT / "data" / "caltech-101-full"
    classes = choose_classes(ensure_categories(extracted))
    if DATA.exists():
        shutil.rmtree(DATA)
    database = DATA / "database"
    queries = DATA / "queries"
    rng = random.Random(SEED)

    ground = {"dataset": "caltech-101-subset", "seed": SEED, "classes": [], "queries": []}
    for name, images in classes:
        ordered = images[:]
        rng.shuffle(ordered)
        db_images = ordered[:DATABASE_PER_CLASS]
        query_images = ordered[DATABASE_PER_CLASS:NEEDED]
        db_rels = []
        for image in db_images:
            dest = database / name / image.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(image, dest)
            db_rels.append(dest.relative_to(DATA).as_posix())
        for image in query_images:
            dest = queries / name / image.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(image, dest)
            ground["queries"].append({
                "path": dest.relative_to(DATA).as_posix(),
                "class": name,
                "relevant": db_rels,
            })
        ground["classes"].append({
            "name": name,
            "database": len(db_images),
            "queries": len(query_images),
        })
        print(f"{name}: {len(db_images)} database, {len(query_images)} queries")

    (DATA / "ground_truth.json").write_text(json.dumps(ground, indent=2))
    print(f"Wrote {DATA / 'ground_truth.json'} ({len(ground['queries'])} queries)")


def category_root_exists(extracted: Path) -> bool:
    return extracted.exists() and any(extracted.rglob("101_ObjectCategories"))


if __name__ == "__main__":
    prepare()
