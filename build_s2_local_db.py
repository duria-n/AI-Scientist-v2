#!/usr/bin/env python3
"""Build a local SQLite search database from downloaded Semantic Scholar datasets."""

from __future__ import annotations

import argparse
from pathlib import Path

from ai_scientist.tools.semantic_scholar import (
    build_local_s2_fts_index,
    get_s2_db_path,
    get_s2_datasets_root,
    ingest_papers_dataset,
    ingest_text_dataset,
    initialize_local_s2_db,
    resolve_release_dir,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a local SQLite Semantic Scholar search DB from datasets."
    )
    parser.add_argument(
        "--datasets-root",
        default=str(get_s2_datasets_root()),
        help="Root directory containing downloaded dataset releases.",
    )
    parser.add_argument(
        "--release-id",
        default="latest",
        help="Release id to ingest, e.g. 2026-03-10 or latest.",
    )
    parser.add_argument(
        "--db-path",
        default=None,
        help="Output SQLite database path. Defaults to <datasets-root>/semantic_scholar.sqlite3",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Delete and rebuild the SQLite database if it already exists.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    datasets_root = Path(args.datasets_root).expanduser().resolve()
    release_dir = resolve_release_dir(datasets_root, args.release_id)
    db_path = get_s2_db_path(args.db_path, datasets_root)

    papers_dir = release_dir / "papers"
    abstracts_dir = release_dir / "abstracts"
    tldrs_dir = release_dir / "tldrs"

    if not papers_dir.exists():
        raise FileNotFoundError(
            f"Missing required dataset directory: {papers_dir}. "
            "Download at least the 'papers' dataset first."
        )

    print(f"Datasets root: {datasets_root}")
    print(f"Release dir: {release_dir}")
    print(f"Database path: {db_path}")

    conn = initialize_local_s2_db(db_path, rebuild=args.rebuild)
    try:
        print("Ingesting papers...")
        paper_count = ingest_papers_dataset(conn, papers_dir, release_dir.name)
        print(f"Inserted papers: {paper_count}")

        if abstracts_dir.exists():
            print("Ingesting abstracts...")
            abstract_count = ingest_text_dataset(conn, abstracts_dir, "abstract")
            print(f"Updated abstracts: {abstract_count}")
        else:
            print("Skipping abstracts: dataset directory not found.")

        if tldrs_dir.exists():
            print("Ingesting TLDRs...")
            tldr_count = ingest_text_dataset(conn, tldrs_dir, "tldr")
            print(f"Updated TLDRs: {tldr_count}")
        else:
            print("Skipping TLDRs: dataset directory not found.")

        print("Building FTS index...")
        build_local_s2_fts_index(conn)
    finally:
        conn.close()

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
