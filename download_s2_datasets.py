#!/usr/bin/env python3
"""Download Semantic Scholar datasets for a given release.

Examples:
  python download_s2_datasets.py --list --release-id 2026-03-10
  python download_s2_datasets.py --dataset papers --release-id 2026-03-10
  python download_s2_datasets.py --dataset papers,abstracts --dest /Data/s2ag
  python download_s2_datasets.py --dataset all --manifest-only

The script reads the API key from ``--api-key`` or ``S2_API_KEY``.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Iterable
from urllib.parse import urlparse

import requests


DEFAULT_DEST = "/home/adminer/Data/s2_datasets"
DEFAULT_TIMEOUT = 60
CHUNK_SIZE = 1024 * 1024
DEFAULT_API_MIN_INTERVAL = 1.1
DEFAULT_API_MAX_RETRIES = 6
DEFAULT_DOWNLOAD_MAX_RETRIES = 5
_LAST_API_REQUEST_TS = 0.0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="List or download Semantic Scholar datasets for a release."
    )
    parser.add_argument(
        "--release-id",
        default="latest",
        help="Dataset release id such as '2026-03-10' or 'latest'.",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        help=(
            "Dataset name. Repeat the flag or pass comma-separated values. "
            "Use 'all' to download every dataset in the release."
        ),
    )
    parser.add_argument(
        "--dest",
        default=DEFAULT_DEST,
        help=f"Download root directory. Default: {DEFAULT_DEST}",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("S2_API_KEY"),
        help="Semantic Scholar API key. Defaults to S2_API_KEY.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Only list the datasets in the release.",
    )
    parser.add_argument(
        "--manifest-only",
        action="store_true",
        help="Save metadata and URL manifests without downloading files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing local files.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT,
        help=f"HTTP timeout in seconds. Default: {DEFAULT_TIMEOUT}",
    )
    parser.add_argument(
        "--api-min-interval",
        type=float,
        default=DEFAULT_API_MIN_INTERVAL,
        help=(
            "Minimum interval between Semantic Scholar API metadata requests in seconds. "
            f"Default: {DEFAULT_API_MIN_INTERVAL}"
        ),
    )
    parser.add_argument(
        "--api-max-retries",
        type=int,
        default=DEFAULT_API_MAX_RETRIES,
        help=f"Maximum retries for 429/5xx API responses. Default: {DEFAULT_API_MAX_RETRIES}",
    )
    parser.add_argument(
        "--download-max-retries",
        type=int,
        default=DEFAULT_DOWNLOAD_MAX_RETRIES,
        help=(
            "Maximum retries for each dataset file download on request/network errors. "
            f"Default: {DEFAULT_DOWNLOAD_MAX_RETRIES}"
        ),
    )
    return parser


def create_session(api_key: str | None) -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": "AI-Scientist-v2/s2-dataset-downloader"})
    if api_key:
        session.headers["x-api-key"] = api_key
    return session


def api_get(
    session: requests.Session,
    url: str,
    *,
    timeout: int,
    min_interval: float,
    max_retries: int,
) -> requests.Response:
    global _LAST_API_REQUEST_TS

    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        elapsed = time.monotonic() - _LAST_API_REQUEST_TS
        wait_time = max(0.0, min_interval - elapsed)
        if wait_time > 0:
            time.sleep(wait_time)

        _LAST_API_REQUEST_TS = time.monotonic()
        try:
            response = session.get(url, timeout=timeout)
        except requests.RequestException as exc:
            last_error = exc
            if attempt >= max_retries:
                raise
            backoff_wait = min(30.0, max(min_interval, 2**attempt))
            print(
                f"API request failed ({exc}). Retrying in {backoff_wait:.1f}s "
                f"[attempt {attempt + 1}/{max_retries}]..."
            )
            time.sleep(backoff_wait)
            continue

        if response.status_code < 400:
            return response

        if response.status_code in {429, 500, 502, 503, 504} and attempt < max_retries:
            retry_after_header = response.headers.get("Retry-After")
            retry_after = None
            if retry_after_header:
                try:
                    retry_after = float(retry_after_header)
                except ValueError:
                    retry_after = None

            backoff_wait = retry_after if retry_after is not None else min(
                60.0, max(min_interval, 2**attempt)
            )
            preview = response.text[:200].replace("\n", " ")
            print(
                f"API returned {response.status_code}. Retrying in {backoff_wait:.1f}s "
                f"[attempt {attempt + 1}/{max_retries}]... Response: {preview}"
            )
            time.sleep(backoff_wait)
            continue

        response.raise_for_status()

    if last_error is not None:
        raise last_error
    raise RuntimeError("API request retry loop exited unexpectedly.")


def get_release_metadata(
    session: requests.Session,
    release_id: str,
    timeout: int,
    min_interval: float,
    max_retries: int,
) -> dict:
    url = f"https://api.semanticscholar.org/datasets/v1/release/{release_id}"
    return api_get(
        session,
        url,
        timeout=timeout,
        min_interval=min_interval,
        max_retries=max_retries,
    ).json()


def get_dataset_metadata(
    session: requests.Session,
    release_id: str,
    dataset_name: str,
    timeout: int,
    min_interval: float,
    max_retries: int,
) -> dict:
    url = (
        "https://api.semanticscholar.org/datasets/v1/"
        f"release/{release_id}/dataset/{dataset_name}"
    )
    return api_get(
        session,
        url,
        timeout=timeout,
        min_interval=min_interval,
        max_retries=max_retries,
    ).json()


def expand_dataset_args(dataset_args: list[str] | None) -> list[str]:
    names: list[str] = []
    if not dataset_args:
        return names

    for item in dataset_args:
        for part in item.split(","):
            part = part.strip()
            if part:
                names.append(part)
    return names


def resolve_dataset_names(requested: list[str], release_meta: dict) -> list[str]:
    available = [dataset["name"] for dataset in release_meta.get("datasets", [])]
    if not requested:
        raise ValueError("No dataset selected. Use --dataset NAME or --list.")

    if any(name.lower() == "all" for name in requested):
        return available

    missing = [name for name in requested if name not in available]
    if missing:
        raise ValueError(
            f"Unknown datasets: {', '.join(missing)}. "
            f"Available datasets: {', '.join(available)}"
        )
    return requested


def summarize_description(text: str) -> str:
    if not text:
        return ""
    return text.strip().splitlines()[0]


def print_release_listing(release_meta: dict) -> None:
    release_id = release_meta.get("release_id", "<unknown>")
    datasets = release_meta.get("datasets", [])
    print(f"Release: {release_id}")
    print(f"Datasets: {len(datasets)}")
    for dataset in datasets:
        desc = summarize_description(dataset.get("description", ""))
        print(f"- {dataset['name']}: {desc}")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def extract_file_entries(dataset_meta: dict) -> list[dict[str, str]]:
    files = dataset_meta.get("files", [])
    entries: list[dict[str, str]] = []
    for index, file_info in enumerate(files, start=1):
        if isinstance(file_info, str):
            url = file_info
            name = ""
        elif isinstance(file_info, dict):
            url = str(file_info.get("url", "")).strip()
            name = str(file_info.get("name", "")).strip()
        else:
            continue

        if not url:
            continue
        entries.append({"index": str(index), "url": url, "name": name})
    return entries


def infer_filename(entry: dict[str, str]) -> str:
    if entry.get("name"):
        return entry["name"]

    parsed = urlparse(entry["url"])
    basename = Path(parsed.path).name
    if basename:
        return basename
    return f"part-{entry['index']}.bin"


def save_manifest(path: Path, entries: Iterable[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(f"{entry['url']}\n")


def download_file(
    session: requests.Session,
    url: str,
    dest_path: Path,
    *,
    timeout: int,
    max_retries: int,
    overwrite: bool,
) -> None:
    if dest_path.exists() and not overwrite:
        print(f"Skip existing file: {dest_path}")
        return

    tmp_path = dest_path.with_suffix(dest_path.suffix + ".part")
    if tmp_path.exists():
        tmp_path.unlink()

    for attempt in range(max_retries + 1):
        if tmp_path.exists():
            tmp_path.unlink()

        try:
            print(f"Downloading: {dest_path.name}")
            with session.get(url, stream=True, timeout=timeout) as response:
                if response.status_code in {429, 500, 502, 503, 504}:
                    response.raise_for_status()

                response.raise_for_status()
                total_bytes = int(response.headers.get("content-length", "0") or 0)
                written = 0
                next_report = 128 * 1024 * 1024

                with tmp_path.open("wb") as handle:
                    for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                        if not chunk:
                            continue
                        handle.write(chunk)
                        written += len(chunk)
                        if total_bytes and written >= next_report:
                            print(
                                f"  progress: {written / (1024 ** 3):.2f}GB / "
                                f"{total_bytes / (1024 ** 3):.2f}GB"
                            )
                            next_report += 128 * 1024 * 1024
            break
        except requests.RequestException as exc:
            if attempt >= max_retries:
                if tmp_path.exists():
                    tmp_path.unlink()
                raise
            retry_after_header = None
            if hasattr(exc, "response") and exc.response is not None:
                retry_after_header = exc.response.headers.get("Retry-After")
            retry_after = None
            if retry_after_header:
                try:
                    retry_after = float(retry_after_header)
                except ValueError:
                    retry_after = None
            backoff_wait = retry_after if retry_after is not None else min(
                60.0, 2 ** attempt
            )
            print(
                f"Download failed for {dest_path.name} ({exc}). Retrying in "
                f"{backoff_wait:.1f}s [attempt {attempt + 1}/{max_retries}]..."
            )
            time.sleep(backoff_wait)

    tmp_path.replace(dest_path)
    if total_bytes:
        print(
            f"Finished: {dest_path} "
            f"({total_bytes / (1024 ** 3):.2f}GB)"
        )
    else:
        print(f"Finished: {dest_path}")


def download_dataset(
    session: requests.Session,
    release_id: str,
    dataset_name: str,
    dest_root: Path,
    *,
    timeout: int,
    api_min_interval: float,
    api_max_retries: int,
    download_max_retries: int,
    manifest_only: bool,
    overwrite: bool,
) -> None:
    print(f"\n=== {dataset_name} ===")
    dataset_meta = get_dataset_metadata(
        session,
        release_id,
        dataset_name,
        timeout,
        api_min_interval,
        api_max_retries,
    )
    entries = extract_file_entries(dataset_meta)

    dataset_dir = dest_root / release_id / dataset_name
    ensure_dir(dataset_dir)
    save_json(dataset_dir / "metadata.json", dataset_meta)
    save_manifest(dataset_dir / "urls.txt", entries)

    print(f"Saved metadata: {dataset_dir / 'metadata.json'}")
    print(f"Saved manifest: {dataset_dir / 'urls.txt'}")
    print(f"Files in manifest: {len(entries)}")
    metadata_cooldown = random.uniform(1.3, 2.0)
    print(f"Cooling down {metadata_cooldown:.1f}s after dataset metadata request...")
    time.sleep(metadata_cooldown)

    if manifest_only:
        return

    for entry in entries:
        filename = infer_filename(entry)
        download_file(
            session,
            entry["url"],
            dataset_dir / filename,
            timeout=timeout,
            max_retries=download_max_retries,
            overwrite=overwrite,
        )


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if not args.list and not args.api_key:
        parser.error("Downloading requires --api-key or S2_API_KEY.")

    session = create_session(args.api_key)
    try:
        release_meta = get_release_metadata(
            session,
            args.release_id,
            args.timeout,
            args.api_min_interval,
            args.api_max_retries,
        )
    except requests.HTTPError as exc:
        print(f"Failed to fetch release metadata: {exc}", file=sys.stderr)
        return 1

    if args.list:
        print_release_listing(release_meta)
        return 0

    try:
        datasets = resolve_dataset_names(
            expand_dataset_args(args.dataset),
            release_meta,
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    dest_root = Path(args.dest).expanduser().resolve()
    ensure_dir(dest_root)
    print(f"Release: {release_meta.get('release_id', args.release_id)}")
    print(f"Destination: {dest_root}")
    print(f"Datasets: {', '.join(datasets)}")
    print(
        f"API pacing: min_interval={args.api_min_interval:.1f}s, "
        f"max_retries={args.api_max_retries}"
    )
    print(f"Download retries per file: {args.download_max_retries}")

    failed_datasets: list[str] = []
    for dataset_name in datasets:
        try:
            download_dataset(
                session,
                release_meta.get("release_id", args.release_id),
                dataset_name,
                dest_root,
                timeout=args.timeout,
                api_min_interval=args.api_min_interval,
                api_max_retries=args.api_max_retries,
                download_max_retries=args.download_max_retries,
                manifest_only=args.manifest_only,
                overwrite=args.overwrite,
            )
        except requests.HTTPError as exc:
            print(f"Failed dataset {dataset_name}: {exc}", file=sys.stderr)
            failed_datasets.append(dataset_name)
        except requests.RequestException as exc:
            print(f"Failed dataset {dataset_name}: {exc}", file=sys.stderr)
            failed_datasets.append(dataset_name)

    if failed_datasets:
        print(
            "\nCompleted with failures. Failed datasets: "
            + ", ".join(failed_datasets),
            file=sys.stderr,
        )
        return 3

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
