# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "httpx>=0.27.0",
#   "python-dateutil>=2.9.0",
# ]
# ///
from __future__ import annotations

import argparse
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib.parse import urlparse

import httpx
from dateutil import parser as date_parser
from dateutil import tz


def normalize_base_url(raw: str) -> str:
    raw = raw.strip().rstrip("/")
    parsed = urlparse(raw)
    if not parsed.scheme:
        raise ValueError("IMMICH_API_URL must include scheme, e.g. https://host")
    # If the path doesn't already end with /api, append it
    if not parsed.path.rstrip("/").endswith("/api"):
        raw = raw + "/api"
    return raw


def load_config() -> Tuple[str, str]:
    base_url = os.environ["IMMICH_API_URL"]
    api_key = os.environ["IMMICH_API_KEY"]
    return normalize_base_url(base_url), api_key


def parse_date_arg(value: Optional[str]) -> Optional[datetime]:
    if value is None:
        return None
    dt = date_parser.parse(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=tz.tzlocal())
    return dt


class ImmichClient:
    def __init__(self, base_url: str, api_key: str, timeout: float = 30.0) -> None:
        self.base_url = base_url.rstrip("/")
        self._client = httpx.Client(
            base_url=self.base_url,
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json",
                "x-api-key": api_key,
            },
            timeout=timeout,
        )

    def close(self) -> None:
        self._client.close()

    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        if not path.startswith("/"):
            path = "/" + path
        resp = self._client.request(method, path, **kwargs)
        resp.raise_for_status()
        if resp.status_code == 204 or not resp.content:
            return None
        return resp.json()

    # --- Albums ---------------------------------------------------------

    def create_album(self, name: str, description: Optional[str] = None) -> Dict[str, Any]:
        body: Dict[str, Any] = {"albumName": name}
        if description:
            body["description"] = description
        return self._request("POST", "/albums", json=body)

    def add_assets_to_album(
        self,
        album_id: str,
        asset_ids: List[str],
        chunk_size: int = 500,
    ) -> None:
        for i in range(0, len(asset_ids), chunk_size):
            chunk = asset_ids[i : i + chunk_size]
            if not chunk:
                continue
            self._request("PUT", f"/albums/{album_id}/assets", json={"ids": chunk})

    # --- People ---------------------------------------------------------

    def get_people(self) -> List[Dict[str, Any]]:
        data = self._request("GET", "/people")
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            if isinstance(data.get("people"), list):
                return data["people"]
            if isinstance(data.get("items"), list):
                return data["items"]
        raise ValueError("Unexpected shape from /people")

    # --- Search ---------------------------------------------------------

    def search_assets_for_person(
        self,
        person_id: str,
        taken_after: Optional[datetime],
        taken_before: Optional[datetime],
        media_type: str = "IMAGE",
        page_size: int = 500,
    ) -> Set[str]:
        asset_ids: Set[str] = set()
        page = 1

        while True:
            body: Dict[str, Any] = {
                "personIds": [person_id],
                "type": media_type,
                "page": page,
                "size": page_size,
                "withDeleted": False,
                "withArchived": True,
            }
            if taken_after is not None:
                body["takenAfter"] = taken_after.astimezone(tz.UTC).isoformat()
            if taken_before is not None:
                body["takenBefore"] = taken_before.astimezone(tz.UTC).isoformat()

            data = self._request("POST", "/search/metadata", json=body)
            assets = data["assets"]
            items = assets["items"]
            if not items:
                break

            for asset in items:
                asset_id = asset.get("id")
                if asset_id:
                    asset_ids.add(asset_id)

            # Simple pagination: stop when we get less than page_size
            if assets.get("count", 0) < page_size:
                break

            page += 1

        return asset_ids


def build_person_lookup(
    people: List[Dict[str, Any]],
) -> Dict[str, List[str]]:
    lookup: Dict[str, List[str]] = {}
    for p in people:
        name = (p.get("name") or "").strip()
        pid = p.get("id")
        if not name or not pid:
            continue
        key = name.lower()
        lookup.setdefault(key, []).append(pid)
    return lookup


def match_people_by_name(
    lookup: Dict[str, List[str]],
    requested_names: List[str],
) -> Tuple[Dict[str, List[str]], List[str]]:
    name_to_ids: Dict[str, List[str]] = {}
    unknown: List[str] = []

    for raw_name in requested_names:
        key = raw_name.strip().lower()
        ids = lookup.get(key)
        if not ids:
            unknown.append(raw_name)
        else:
            name_to_ids[raw_name] = ids

    return name_to_ids, unknown


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create an Immich album and populate it with all images "
            "containing any of the specified people within an optional "
            "takenAfter/takenBefore date range."
        )
    )

    parser.add_argument(
        "album_name",
        help="Name of the album to create.",
    )
    parser.add_argument(
        "--taken_after",
        dest="taken_after",
        metavar="DATETIME",
        help="Only include assets taken at or after this time.",
    )
    parser.add_argument(
        "--taken_before",
        dest="taken_before",
        metavar="DATETIME",
        help="Only include assets taken before this time.",
    )
    parser.add_argument(
        "-p",
        "--person",
        dest="people",
        action="append",
        default=[],
        help="Person name (can be given multiple times).",
    )

    args = parser.parse_args(argv)

    if not args.people:
        parser.error("You must provide at least one -p/--person name.")

    if args.taken_after and args.taken_before:
        ta = parse_date_arg(args.taken_after)
        tb = parse_date_arg(args.taken_before)
        if ta is not None and tb is not None and ta > tb:
            parser.error("taken-after must be <= taken-before.")

    return args


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    taken_after = parse_date_arg(args.taken_after) if args.taken_after else None
    taken_before = parse_date_arg(args.taken_before) if args.taken_before else None

    base_url, api_key = load_config()
    client = ImmichClient(base_url, api_key)

    try:
        album = client.create_album(args.album_name)
        album_id = album["id"]
        album_name_actual = album.get("albumName", args.album_name)
        print(f"Created album '{album_name_actual}' ({album_id})")

        people = client.get_people()
        lookup = build_person_lookup(people)
        name_to_ids, unknown_names = match_people_by_name(lookup, args.people)

        if unknown_names:
            print(
                "Warning: no people found for: "
                + ", ".join(repr(n) for n in unknown_names)
            )

        if not name_to_ids:
            print("No matching people; album remains empty.")
            return 0

        all_asset_ids: Set[str] = set()
        for person_name, person_ids in name_to_ids.items():
            print(
                f"Searching assets for '{person_name}' "
                f"(IDs: {', '.join(person_ids)})..."
            )
            for pid in person_ids:
                ids_for_person = client.search_assets_for_person(
                    pid,
                    taken_after=taken_after,
                    taken_before=taken_before,
                )
                print(f"  {len(ids_for_person)} assets for person-id {pid}")
                all_asset_ids.update(ids_for_person)

        if not all_asset_ids:
            print("No matching assets found; album remains empty.")
            return 0

        ids_list = sorted(all_asset_ids)
        print(f"Adding {len(ids_list)} assets to album {album_id}...")
        client.add_assets_to_album(album_id, ids_list)

        print(
            f"Done. Album '{album_name_actual}' ({album_id}) now has "
            f"{len(ids_list)} added assets."
        )
        if unknown_names:
            print(
                "Names not found (no assets added for them): "
                + ", ".join(repr(n) for n in unknown_names)
            )
        return 0
    finally:
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())