#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11,<3.12"
# dependencies = [
#   "requests>=2.32",
#   "python-dateutil>=2.9",
#   "Pillow>=10.0",
#   "opencv-python-headless>=4.10",
#   "numpy>=1.26",
#   "imagehash>=4.3",
#   "simple-aesthetics-predictor==0.1.2",
#   "deepface>=0.0.92",
#   "tqdm>=4.66",
# ]
# ///

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

import requests
from dateutil import parser as dateparser
from PIL import Image, UnidentifiedImageError
import numpy as np
import cv2
import imagehash
from simple_aesthetics_predictor import AestheticsPredictor
from deepface import DeepFace
from tqdm import tqdm


# ---------------- Immich client ----------------


class ImmichClient:
    def __init__(self, base_url: str, api_key: str):
        # Expect something like "https://host/api"
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.session.headers.update(
            {
                "x-api-key": api_key,
                "Accept": "application/json",
            }
        )

    def _url(self, path: str) -> str:
        if not path.startswith("/"):
            path = "/" + path
        return self.base_url + path

    def get_album(self, album_id: str) -> Dict[str, Any]:
        resp = self.session.get(self._url(f"/albums/{album_id}"))
        resp.raise_for_status()
        return resp.json()

    def get_asset(self, asset_id: str) -> Dict[str, Any]:
        resp = self.session.get(self._url(f"/assets/{asset_id}"))
        resp.raise_for_status()
        return resp.json()

    def create_album(self, name: str, description: Optional[str] = None) -> str:
        payload: Dict[str, Any] = {"albumName": name}
        if description:
            payload["description"] = description
        resp = self.session.post(self._url("/albums"), json=payload)
        resp.raise_for_status()
        data = resp.json()
        return data["id"]

    def add_assets_to_album(self, album_id: str, asset_ids: List[str], chunk_size: int = 500) -> None:
        for i in range(0, len(asset_ids), chunk_size):
            chunk = asset_ids[i : i + chunk_size]
            payload = {"ids": chunk}
            resp = self.session.put(self._url(f"/albums/{album_id}/assets"), json=payload)
            resp.raise_for_status()


# ---------------- Data models ----------------


@dataclass
class AssetFeatures:
    id: str
    path: Path
    dt: Optional[datetime]
    month_index: Optional[int]
    aesthetic_raw: float
    emotion_raw: float
    sharpness_raw: float
    phash: imagehash.ImageHash
    aesthetic: float = 0.0
    emotion: float = 0.0
    sharpness: float = 0.0
    base_score: float = 0.0


# ---------------- Path mapping ----------------


def map_path(original_path: str, immich_prefix: Optional[str], local_prefix: Optional[str]) -> Path:
    p = original_path
    if immich_prefix and local_prefix and p.startswith(immich_prefix):
        rel = p[len(immich_prefix) :].lstrip("/\\")
        return Path(local_prefix) / rel
    return Path(p)


# ---------------- Feature models (lazy globals) ----------------

_aesthetic_model: Optional[AestheticsPredictor] = None


def load_aesthetic_model() -> AestheticsPredictor:
    global _aesthetic_model
    if _aesthetic_model is None:
        _aesthetic_model = AestheticsPredictor.from_pretrained("v2-5")
    return _aesthetic_model


def compute_aesthetic_score(img: Image.Image) -> float:
    model = load_aesthetic_model()
    # Model expects a PIL image; returns a score ~1..10
    score_arr = model.predict(img)
    score = float(score_arr[0])
    # Roughly normalize to [0,1]
    return (score - 1.0) / 9.0


def compute_emotion_score(img: Image.Image) -> float:
    """
    Use DeepFace emotion analysis to get a 'positivity' score in [0,1].
    If no faces or analysis fails, return 0.5 (neutral).
    """
    try:
        # Convert to np array in RGB
        img_np = np.array(img.convert("RGB"))
        # deepface expects BGR by default sometimes, but analyze works with RGB arrays too.
        analysis = DeepFace.analyze(
            img_np,
            actions=["emotion"],
            enforce_detection=False,
            prog_bar=False,
        )
        # DeepFace returns either dict or list of dicts depending on version
        if isinstance(analysis, list):
            faces = analysis
        else:
            faces = [analysis]

        if not faces:
            return 0.5

        scores: List[float] = []
        for face in faces:
            emo = face.get("emotion") or {}
            # Some positive-ish weighting
            happy = float(emo.get("happy", 0.0))
            neutral = float(emo.get("neutral", 0.0))
            surprise = float(emo.get("surprise", 0.0))
            # DeepFace outputs in percentages (0-100)
            pos = happy + 0.5 * surprise + 0.5 * neutral
            pos /= 100.0  # normalize
            pos = max(0.0, min(1.0, pos))
            scores.append(pos)

        if not scores:
            return 0.5
        return float(sum(scores) / len(scores))
    except Exception:
        return 0.5


def compute_sharpness_raw(img: Image.Image) -> float:
    """
    Variance of Laplacian as a sharpness proxy.
    """
    gray = img.convert("L")
    gray_np = np.array(gray)
    lap = cv2.Laplacian(gray_np, cv2.CV_64F)
    return float(lap.var())


def compute_phash(img: Image.Image) -> imagehash.ImageHash:
    img_small = img.copy()
    img_small.thumbnail((256, 256))
    return imagehash.phash(img_small)


# ---------------- Normalization & scoring ----------------


def normalize(values: List[float]) -> List[float]:
    if not values:
        return values
    vmin = min(values)
    vmax = max(values)
    if vmax - vmin < 1e-9:
        # All equal, just return middle
        return [0.5 for _ in values]
    return [(v - vmin) / (vmax - vmin) for v in values]


def compute_base_scores(assets: List[AssetFeatures]) -> None:
    # Collect raw values
    aest_raw = [a.aesthetic_raw for a in assets]
    emo_raw = [a.emotion_raw for a in assets]
    sharp_raw = [a.sharpness_raw for a in assets]

    aest_norm = normalize(aest_raw)
    emo_norm = normalize(emo_raw)
    sharp_norm = normalize(sharp_raw)

    for a, ae, em, sh in zip(assets, aest_norm, emo_norm, sharp_norm):
        a.aesthetic = ae
        a.emotion = em
        a.sharpness = sh
        # Base score without similarity (we'll add that during selection)
        a.base_score = 0.5 * ae + 0.3 * em + 0.1 * sh


# ---------------- Time & month helpers ----------------


def parse_datetime(dt_str: Optional[str]) -> Optional[datetime]:
    if not dt_str:
        return None
    try:
        dt = dateparser.parse(dt_str)
        if dt is not None and dt.tzinfo is not None:
            dt = dt.astimezone(None).replace(tzinfo=None)
        return dt
    except Exception:
        return None


def pick_best_datetime(asset: Dict[str, Any]) -> Optional[datetime]:
    exif = asset.get("exifInfo") or {}
    dt = parse_datetime(exif.get("dateTimeOriginal"))
    if dt:
        return dt

    # Fall back to other fields Immich tends to have
    for key in ["fileCreatedAt", "fileModifiedAt", "createdAt", "updatedAt"]:
        dt = parse_datetime(asset.get(key))
        if dt:
            return dt
    return None


def month_index_for(dt: Optional[datetime], year_filter: Optional[int]) -> Optional[int]:
    if dt is None:
        return None
    if year_filter is not None and dt.year != year_filter:
        return None
    # 0-based month index
    return dt.month - 1


# ---------------- Diversity & selection ----------------


def similarity_penalty(candidate_hash: imagehash.ImageHash, selected_hashes: List[imagehash.ImageHash]) -> float:
    if not selected_hashes:
        return 0.0
    dists = [candidate_hash - h for h in selected_hashes]
    min_dist = min(dists)
    max_bits = candidate_hash.hash.size  # usually 64
    # 1.0 = identical, 0.0 = very different
    penalty = 1.0 - (min_dist / max_bits)
    if penalty < 0.0:
        penalty = 0.0
    if penalty > 1.0:
        penalty = 1.0
    return penalty


def compute_month_quotas(
    assets: List[AssetFeatures], top_n: int
) -> Dict[int, int]:
    # Count candidates per month
    month_counts: Dict[int, int] = {}
    for a in assets:
        if a.month_index is not None:
            month_counts[a.month_index] = month_counts.get(a.month_index, 0) + 1

    if not month_counts:
        return {}

    quotas: Dict[int, int] = {}
    q = top_n / 12.0
    base_total = 0
    for m in range(12):
        if month_counts.get(m, 0) > 0:
            quotas[m] = int(q)
            base_total += quotas[m]

    # Distribute remaining slots favoring months with more candidates
    remaining = max(0, top_n - base_total)
    if remaining > 0:
        # Sort months by available count desc
        months_by_count = sorted(
            [m for m in quotas.keys()],
            key=lambda m: month_counts[m],
            reverse=True,
        )
        idx = 0
        while remaining > 0 and months_by_count:
            m = months_by_count[idx % len(months_by_count)]
            if quotas[m] < month_counts[m]:
                quotas[m] += 1
                remaining -= 1
            idx += 1
            # If all months saturated, break
            if all(quotas[m] >= month_counts[m] for m in months_by_count):
                break

    return quotas


def select_assets(assets: List[AssetFeatures], top_n: int) -> List[AssetFeatures]:
    # Filter out assets without month_index (we'll handle leftovers later)
    with_month = [a for a in assets if a.month_index is not None]
    no_month = [a for a in assets if a.month_index is None]

    # Sort all by base_score desc
    with_month.sort(key=lambda a: a.base_score, reverse=True)
    no_month.sort(key=lambda a: a.base_score, reverse=True)

    quotas = compute_month_quotas(with_month, top_n)

    # Build per-month lists
    per_month: Dict[int, List[AssetFeatures]] = {m: [] for m in range(12)}
    for a in with_month:
        if a.month_index is not None:
            per_month[a.month_index].append(a)

    for m in per_month:
        per_month[m].sort(key=lambda a: a.base_score, reverse=True)

    selected: List[AssetFeatures] = []
    selected_hashes: List[imagehash.ImageHash] = []
    selected_counts: Dict[int, int] = {m: 0 for m in range(12)}

    # First pass: month-balanced round-robin
    while len(selected) < top_n and any(per_month[m] for m in quotas.keys()):
        made_progress = False
        for m in range(12):
            if m not in quotas:
                continue
            if selected_counts[m] >= quotas[m]:
                continue
            if not per_month[m]:
                continue

            cand = per_month[m].pop(0)
            pen = similarity_penalty(cand.phash, selected_hashes)
            adjusted = cand.base_score + 0.1 * (1.0 - pen)

            # Simple rule: accept unless extremely similar
            if pen < 0.7 or not selected:
                cand.base_score = adjusted  # update if you want
                selected.append(cand)
                selected_hashes.append(cand.phash)
                selected_counts[m] += 1
                made_progress = True

            if len(selected) >= top_n:
                break
        if not made_progress:
            break

    # Second pass: fill remaining slots from any month + no-month, highest base_score first
    if len(selected) < top_n:
        remaining_assets = []
        for m in range(12):
            remaining_assets.extend(per_month[m])
        remaining_assets.extend(no_month)
        remaining_assets.sort(key=lambda a: a.base_score, reverse=True)

        for cand in remaining_assets:
            if len(selected) >= top_n:
                break
            pen = similarity_penalty(cand.phash, selected_hashes)
            if pen >= 0.7:
                continue
            adjusted = cand.base_score + 0.1 * (1.0 - pen)
            cand.base_score = adjusted
            selected.append(cand)
            selected_hashes.append(cand.phash)

    return selected[:top_n]


# ---------------- Main pipeline ----------------


def arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Select top N calendar-worthy photos from an Immich album and create a new album."
    )
    p.add_argument("source_album_id", help="UUID of the source album")
    p.add_argument("output_album_name", help="Name of the new output album")
    p.add_argument("top_n", type=int, help="Number of photos to select")

    p.add_argument(
        "--year",
        type=int,
        default=None,
        help="Restrict photos to this year (based on EXIF or fileCreatedAt)",
    )
    p.add_argument(
        "--immich-url",
        default=os.environ.get("IMMICH_URL", "").rstrip("/"),
        help="Base Immich API URL, e.g. https://host/api (or set IMMICH_URL)",
    )
    p.add_argument(
        "--api-key",
        default=os.environ.get("IMMICH_API_KEY", ""),
        help="Immich API key (or set IMMICH_API_KEY)",
    )
    p.add_argument(
        "--immich-path-prefix",
        default=None,
        help="Prefix of originalPath inside Immich container, e.g. /usr/src/app/upload",
    )
    p.add_argument(
        "--local-path-prefix",
        default=None,
        help="Corresponding prefix on host, e.g. /Volumes/whatever/upload",
    )
    return p


def main() -> None:
    parser = arg_parser()
    args = parser.parse_args()

    if not args.immich_url:
        raise SystemExit("You must specify --immich-url or set IMMICH_URL")

    api_key = args.api_key
    if not api_key:
        raise SystemExit("You must specify --api-key or set IMMICH_API_KEY")

    client = ImmichClient(args.immich_url, api_key)

    print(f"Fetching album {args.source_album_id}...")
    album = client.get_album(args.source_album_id)
    album_assets = album.get("assets") or []
    asset_ids = [a["id"] for a in album_assets]

    if not asset_ids:
        raise SystemExit("Source album has no assets (or 'assets' field not present).")

    print(f"Album has {len(asset_ids)} assets; hydrating them via /assets/{id}...")

    hydrated_assets: List[Dict[str, Any]] = []
    for asset_id in tqdm(asset_ids, desc="Fetching assets"):
        try:
            asset = client.get_asset(asset_id)
        except Exception as e:
            print(f"Skipping asset {asset_id} due to error: {e}")
            continue
        hydrated_assets.append(asset)

    features: List[AssetFeatures] = []

    print("Computing scores from local files...")

    for asset in tqdm(hydrated_assets, desc="Analyzing images"):
        asset_id = asset.get("id")
        asset_type = asset.get("type")

        # Skip if no asset ID
        if not asset_id:
            continue

        # Skip non-images
        if asset_type and asset_type.upper() != "IMAGE":
            continue

        original_path = asset.get("originalPath")
        if not original_path:
            continue

        local_path = map_path(original_path, args.immich_path_prefix, args.local_path_prefix)
        if not local_path.is_file():
            # Try originalPath as-is if mapping failed
            if local_path != Path(original_path) and Path(original_path).is_file():
                local_path = Path(original_path)
            else:
                continue

        try:
            with Image.open(local_path) as img:
                # Basic resize for speed
                img = img.convert("RGB")
                img.thumbnail((1024, 1024))

                aest_raw = compute_aesthetic_score(img)
                emo_raw = compute_emotion_score(img)
                sharp_raw = compute_sharpness_raw(img)
                ph = compute_phash(img)
        except (UnidentifiedImageError, OSError):
            continue
        except Exception as e:
            # If some model explodes for one image, just skip it
            print(f"Skipping {local_path} due to error: {e}")
            continue

        dt = pick_best_datetime(asset)
        m_idx = month_index_for(dt, args.year)

        # If year is specified and this asset is not in that year, skip
        if args.year is not None and (dt is None or dt.year != args.year):
            continue

        af = AssetFeatures(
            id=asset_id,
            path=local_path,
            dt=dt,
            month_index=m_idx,
            aesthetic_raw=aest_raw,
            emotion_raw=emo_raw,
            sharpness_raw=sharp_raw,
            phash=ph,
        )
        features.append(af)

    if not features:
        raise SystemExit("No usable image assets found after filtering and analysis.")

    print(f"Computed features for {len(features)} assets. Normalizing scores...")
    compute_base_scores(features)

    top_n = min(args.top_n, len(features))
    print(f"Selecting top {top_n} assets with diversity and time spread...")
    selected = select_assets(features, top_n)

    print(f"Creating new album '{args.output_album_name}'...")
    description = f"Auto-selected calendar album from {args.source_album_id}"
    new_album_id = client.create_album(args.output_album_name, description=description)

    selected_ids = [a.id for a in selected]
    print(f"Adding {len(selected_ids)} assets to new album {new_album_id}...")
    client.add_assets_to_album(new_album_id, selected_ids)

    # Summary
    per_month: Dict[int, int] = {}
    for a in selected:
        if a.month_index is not None:
            per_month[a.month_index] = per_month.get(a.month_index, 0) + 1

    print("\nDone.")
    print(f"New album ID: {new_album_id}")
    print("Selected counts per month (0=Jan ... 11=Dec):")
    for m in sorted(per_month.keys()):
        print(f"  {m}: {per_month[m]} photos")


if __name__ == "__main__":
    main()
