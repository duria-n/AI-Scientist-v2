import gzip
import json
import os
import re
import sqlite3
import time
import warnings
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Union

import backoff
import requests

from ai_scientist.tools.base_tool import BaseTool

SEMANTIC_SCHOLAR_SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
SEMANTIC_SCHOLAR_TIMEOUT_SECS = 20
DEFAULT_TOOL_FIELDS = "title,authors,venue,year,abstract,citationCount"
DEFAULT_WRITEUP_FIELDS = (
    "title,authors,venue,year,abstract,citationStyles,citationCount"
)
DEFAULT_S2_DATASETS_ROOT = Path("/media/adminer/Expansion/s2_datasets")
DEFAULT_S2_DB_FILENAME = "semantic_scholar.sqlite3"
DEFAULT_S2_SEARCH_BACKEND = "auto"
RELEASE_ID_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")
IGNORED_DATASET_FILES = {"metadata.json", "urls.txt"}
TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+")
VALID_BACKENDS = {"auto", "local", "api"}


def on_backoff(details: Dict) -> None:
    print(
        f"Backing off {details['wait']:0.1f} seconds after {details['tries']} tries "
        f"calling function {details['target'].__name__} at {time.strftime('%X')}"
    )


def get_s2_search_backend(backend: Optional[str] = None) -> str:
    backend_name = backend or os.getenv("S2_SEARCH_BACKEND", DEFAULT_S2_SEARCH_BACKEND)
    backend_name = backend_name.strip().lower()
    if backend_name not in VALID_BACKENDS:
        raise ValueError(
            f"Invalid S2 search backend: {backend_name!r}. "
            f"Expected one of: {', '.join(sorted(VALID_BACKENDS))}"
        )
    return backend_name


def _load_s2_api_key() -> Optional[str]:
    api_key = os.getenv("S2_API_KEY")
    if api_key is None:
        return None

    stripped_api_key = api_key.strip()
    if not stripped_api_key:
        return None

    if stripped_api_key != api_key:
        warnings.warn(
            "S2_API_KEY contains leading or trailing whitespace; stripping it before use."
        )

    return stripped_api_key


def _build_headers(api_key: Optional[str]) -> Dict[str, str]:
    headers = {
        "Accept": "application/json",
        "User-Agent": "AI-Scientist-v2/semantic-scholar",
    }
    if api_key:
        headers["X-API-KEY"] = api_key
    return headers


def _raise_for_semantic_scholar_error(
    rsp: requests.Response, used_api_key: bool
) -> None:
    body_preview = rsp.text[:500]
    details = [f"Semantic Scholar search failed with status {rsp.status_code}."]

    if rsp.status_code == 403 and used_api_key:
        details.append(
            "The supplied S2_API_KEY was rejected. Check that the key is active and "
            "does not contain hidden whitespace."
        )
    elif rsp.status_code == 429:
        details.append(
            "Semantic Scholar rate limited the request. Retry later or use a valid API key."
        )

    details.append(f"Response preview: {body_preview}")
    raise requests.exceptions.HTTPError(" ".join(details), response=rsp)


def _request_semantic_scholar_search(
    query: str, result_limit: int, fields: str, api_key: Optional[str]
) -> requests.Response:
    return requests.get(
        SEMANTIC_SCHOLAR_SEARCH_URL,
        headers=_build_headers(api_key),
        params={
            "query": query,
            "limit": result_limit,
            "fields": fields,
        },
        timeout=SEMANTIC_SCHOLAR_TIMEOUT_SECS,
    )


def _search_for_papers_via_api(
    query: str, result_limit: int, fields: str
) -> Optional[List[Dict]]:
    if not query:
        return None

    api_key = _load_s2_api_key()
    if not api_key:
        warnings.warn(
            "No Semantic Scholar API key found. Requests will be subject to stricter rate limits. "
            "Set the S2_API_KEY environment variable for higher limits."
        )

    used_api_key = bool(api_key)
    rsp = _request_semantic_scholar_search(query, result_limit, fields, api_key)

    if rsp.status_code == 403 and api_key:
        warnings.warn(
            "Semantic Scholar rejected S2_API_KEY with 403; retrying without the key."
        )
        used_api_key = False
        rsp = _request_semantic_scholar_search(query, result_limit, fields, None)

    if rsp.status_code >= 400:
        _raise_for_semantic_scholar_error(rsp, used_api_key=used_api_key)

    results = rsp.json()
    total = results.get("total", 0)
    if total == 0:
        return None

    return results.get("data", [])


def get_s2_datasets_root(root: Optional[Union[str, Path]] = None) -> Path:
    if root is not None:
        return Path(root).expanduser().resolve()

    env_root = os.getenv("S2_DATASETS_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()

    return DEFAULT_S2_DATASETS_ROOT


def get_s2_db_path(
    db_path: Optional[Union[str, Path]] = None,
    datasets_root: Optional[Union[str, Path]] = None,
) -> Path:
    if db_path is not None:
        return Path(db_path).expanduser().resolve()

    env_db_path = os.getenv("S2_LOCAL_DB_PATH")
    if env_db_path:
        return Path(env_db_path).expanduser().resolve()

    return get_s2_datasets_root(datasets_root) / DEFAULT_S2_DB_FILENAME


def resolve_release_dir(
    datasets_root: Optional[Union[str, Path]] = None,
    release_id: Optional[str] = None,
) -> Path:
    root = get_s2_datasets_root(datasets_root)
    if not root.exists():
        raise FileNotFoundError(f"Datasets root does not exist: {root}")

    if release_id and release_id != "latest":
        release_dir = root / release_id
        if not release_dir.exists():
            raise FileNotFoundError(f"Release directory does not exist: {release_dir}")
        return release_dir

    release_dirs = sorted(
        path for path in root.iterdir() if path.is_dir() and RELEASE_ID_PATTERN.match(path.name)
    )
    if not release_dirs:
        raise FileNotFoundError(
            f"No release directories found under {root}. "
            "Download datasets first with download_s2_datasets.py."
        )
    return release_dirs[-1]


def iter_dataset_files(dataset_dir: Path) -> Iterable[Path]:
    if not dataset_dir.exists():
        return []

    return sorted(
        path
        for path in dataset_dir.rglob("*")
        if path.is_file()
        and path.name not in IGNORED_DATASET_FILES
        and not path.name.endswith(".part")
    )


def open_dataset_text_file(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def iter_jsonl_records(dataset_dir: Path) -> Iterable[dict]:
    for file_path in iter_dataset_files(dataset_dir):
        with open_dataset_text_file(file_path) as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(record, dict):
                    yield record


def _normalize_key(key: str) -> str:
    return "".join(ch for ch in key.lower() if ch.isalnum())


def _get_case_insensitive(mapping: dict, key: str):
    target = _normalize_key(key)
    for existing_key, value in mapping.items():
        if _normalize_key(str(existing_key)) == target:
            return value
    return None


def _get_nested(record: Optional[dict], *path: str):
    current = record
    for key in path:
        if not isinstance(current, dict):
            return None
        current = _get_case_insensitive(current, key)
        if current is None:
            return None
    return current


def _coerce_int(value) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _coerce_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def _extract_corpus_id(record: dict) -> Optional[str]:
    for path in (
        ("corpusId",),
        ("corpusid",),
        ("paper", "corpusId"),
        ("paper", "corpusid"),
    ):
        value = _get_nested(record, *path)
        if value is not None and value != "":
            return str(value)
    return None


def _extract_paper_id(record: dict) -> str:
    for path in (
        ("paperId",),
        ("paperid",),
        ("paper", "paperId"),
        ("paper", "paperid"),
        ("sha",),
    ):
        value = _get_nested(record, *path)
        if value:
            return str(value)
    return ""


def _extract_title(record: dict) -> str:
    for path in (("title",), ("paper", "title")):
        value = _get_nested(record, *path)
        if value:
            return _coerce_text(value)
    return ""


def _extract_authors(record: dict) -> List[Dict[str, str]]:
    raw_authors = (
        _get_nested(record, "authors")
        or _get_nested(record, "paper", "authors")
        or []
    )
    authors: List[Dict[str, str]] = []
    if not isinstance(raw_authors, list):
        return authors

    for author in raw_authors:
        if isinstance(author, str):
            name = author.strip()
        elif isinstance(author, dict):
            name = _coerce_text(
                _get_nested(author, "name")
                or _get_nested(author, "author", "name")
                or _get_nested(author, "authorName")
            )
        else:
            name = ""

        if name:
            authors.append({"name": name})
    return authors


def _extract_venue(record: dict) -> str:
    for path in (
        ("venue",),
        ("publicationVenue", "name"),
        ("publicationvenue", "name"),
        ("journal", "name"),
        ("booktitle",),
    ):
        value = _get_nested(record, *path)
        if value:
            return _coerce_text(value)
    return ""


def _extract_year(record: dict) -> Optional[int]:
    for path in (("year",), ("publicationDate", "year"), ("publicationdate", "year")):
        value = _get_nested(record, *path)
        year = _coerce_int(value)
        if year is not None:
            return year
    return None


def _extract_citation_count(record: dict) -> int:
    for path in (("citationCount",), ("citationcount",)):
        value = _get_nested(record, *path)
        citation_count = _coerce_int(value)
        if citation_count is not None:
            return citation_count
    return 0


def _extract_external_ids(record: dict) -> dict:
    external_ids = (
        _get_nested(record, "externalIds")
        or _get_nested(record, "externalids")
        or _get_nested(record, "openAccessInfo", "externalIds")
        or _get_nested(record, "openaccessinfo", "externalids")
        or _get_nested(record, "paper", "externalIds")
        or {}
    )
    return external_ids if isinstance(external_ids, dict) else {}


def _extract_doi(record: dict) -> str:
    external_ids = _extract_external_ids(record)
    return _coerce_text(
        _get_case_insensitive(external_ids, "DOI")
        or _get_case_insensitive(external_ids, "doi")
    )


def _extract_arxiv_id(record: dict) -> str:
    external_ids = _extract_external_ids(record)
    return _coerce_text(
        _get_case_insensitive(external_ids, "ArXiv")
        or _get_case_insensitive(external_ids, "arxiv")
    )


def _extract_abstract(record: dict) -> str:
    for path in (
        ("abstract",),
        ("text",),
        ("paper", "abstract"),
        ("summary",),
    ):
        value = _get_nested(record, *path)
        if isinstance(value, str) and value.strip():
            return value.strip()

    abstract = _get_nested(record, "abstract")
    if isinstance(abstract, dict):
        for key in ("text", "value"):
            value = _get_nested(abstract, key)
            if isinstance(value, str) and value.strip():
                return value.strip()

    return ""


def _extract_tldr(record: dict) -> str:
    for path in (("text",), ("tldr",), ("summary",)):
        value = _get_nested(record, *path)
        if isinstance(value, str) and value.strip():
            return value.strip()

    tldr = _get_nested(record, "tldr")
    if isinstance(tldr, dict):
        for key in ("text", "value"):
            value = _get_nested(tldr, key)
            if isinstance(value, str) and value.strip():
                return value.strip()

    return ""


def _build_cite_key(authors: List[Dict[str, str]], year: Optional[int], title: str) -> str:
    first_author = authors[0]["name"] if authors else "paper"
    last_name = TOKEN_PATTERN.findall(first_author.lower())
    title_token = TOKEN_PATTERN.findall(title.lower())
    base = (last_name[-1] if last_name else "paper") + str(year or "undated")
    if title_token:
        base += title_token[0]
    return base


def _escape_bibtex_value(value: str) -> str:
    return value.replace("\\", "\\\\").replace("{", "\\{").replace("}", "\\}")


def _build_bibtex(record: dict) -> str:
    authors = record.get("authors", [])
    title = _coerce_text(record.get("title", "Untitled"))
    venue = _coerce_text(record.get("venue", ""))
    year = record.get("year")
    doi = _coerce_text(record.get("doi", ""))
    arxiv_id = _coerce_text(record.get("arxivId", ""))
    cite_key = _build_cite_key(authors, year, title)
    author_string = " and ".join(author["name"] for author in authors if author.get("name"))

    entry_type = "article" if venue else "misc"
    fields = [
        ("title", title),
        ("author", author_string),
        ("year", str(year) if year else ""),
        ("journal", venue if entry_type == "article" else ""),
        ("note", f"arXiv:{arxiv_id}" if arxiv_id else ""),
        ("doi", doi),
    ]

    lines = [f"@{entry_type}{{{cite_key},"]
    for field_name, field_value in fields:
        if field_value:
            lines.append(
                f"  {field_name} = {{{_escape_bibtex_value(field_value)}}},"
            )
    lines.append("}")
    return "\n".join(lines)


def initialize_local_s2_db(db_path: Union[str, Path], rebuild: bool = False) -> sqlite3.Connection:
    db_file = Path(db_path).expanduser().resolve()
    db_file.parent.mkdir(parents=True, exist_ok=True)

    if rebuild and db_file.exists():
        db_file.unlink()
    elif db_file.exists():
        raise FileExistsError(
            f"Database already exists: {db_file}. Re-run with --rebuild to overwrite it."
        )

    conn = sqlite3.connect(db_file)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA temp_store=MEMORY")
    conn.execute("PRAGMA cache_size=-200000")

    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS papers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            corpus_id TEXT UNIQUE NOT NULL,
            paper_id TEXT,
            title TEXT NOT NULL DEFAULT '',
            authors_json TEXT NOT NULL DEFAULT '[]',
            authors_text TEXT NOT NULL DEFAULT '',
            venue TEXT NOT NULL DEFAULT '',
            year INTEGER,
            citation_count INTEGER NOT NULL DEFAULT 0,
            abstract TEXT NOT NULL DEFAULT '',
            tldr TEXT NOT NULL DEFAULT '',
            doi TEXT NOT NULL DEFAULT '',
            arxiv_id TEXT NOT NULL DEFAULT '',
            source_release TEXT NOT NULL DEFAULT ''
        );
        CREATE INDEX IF NOT EXISTS idx_papers_corpus_id ON papers(corpus_id);
        CREATE INDEX IF NOT EXISTS idx_papers_citation_count ON papers(citation_count DESC);
        CREATE INDEX IF NOT EXISTS idx_papers_year ON papers(year DESC);
        """
    )
    return conn


def ingest_papers_dataset(
    conn: sqlite3.Connection, papers_dir: Path, source_release: str
) -> int:
    insert_sql = """
        INSERT INTO papers (
            corpus_id,
            paper_id,
            title,
            authors_json,
            authors_text,
            venue,
            year,
            citation_count,
            doi,
            arxiv_id,
            source_release
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """

    batch = []
    inserted = 0
    for record in iter_jsonl_records(papers_dir):
        corpus_id = _extract_corpus_id(record)
        if not corpus_id:
            continue

        authors = _extract_authors(record)
        batch.append(
            (
                corpus_id,
                _extract_paper_id(record),
                _extract_title(record),
                json.dumps(authors, ensure_ascii=False),
                ", ".join(author["name"] for author in authors),
                _extract_venue(record),
                _extract_year(record),
                _extract_citation_count(record),
                _extract_doi(record),
                _extract_arxiv_id(record),
                source_release,
            )
        )

        if len(batch) >= 1000:
            conn.executemany(insert_sql, batch)
            conn.commit()
            inserted += len(batch)
            batch.clear()

    if batch:
        conn.executemany(insert_sql, batch)
        conn.commit()
        inserted += len(batch)

    return inserted


def ingest_text_dataset(
    conn: sqlite3.Connection, dataset_dir: Path, text_column: str
) -> int:
    assert text_column in {"abstract", "tldr"}
    update_sql = f"""
        UPDATE papers
        SET {text_column} = ?
        WHERE corpus_id = ?
    """

    batch = []
    updated = 0
    for record in iter_jsonl_records(dataset_dir):
        corpus_id = _extract_corpus_id(record)
        if not corpus_id:
            continue

        text_value = (
            _extract_abstract(record) if text_column == "abstract" else _extract_tldr(record)
        )
        if not text_value:
            continue

        batch.append((text_value, corpus_id))
        if len(batch) >= 1000:
            conn.executemany(update_sql, batch)
            conn.commit()
            updated += len(batch)
            batch.clear()

    if batch:
        conn.executemany(update_sql, batch)
        conn.commit()
        updated += len(batch)

    return updated


def build_local_s2_fts_index(conn: sqlite3.Connection) -> None:
    conn.execute("DROP TABLE IF EXISTS papers_fts")
    try:
        conn.execute(
            """
            CREATE VIRTUAL TABLE papers_fts USING fts5(
                title,
                abstract,
                tldr,
                authors_text,
                venue,
                content='',
                tokenize='porter unicode61 remove_diacritics 1'
            )
            """
        )
    except sqlite3.OperationalError as exc:
        raise RuntimeError(
            "Failed to create the SQLite FTS5 index. "
            "Your Python sqlite3 build may not include FTS5 support."
        ) from exc
    conn.execute(
        """
        INSERT INTO papers_fts(rowid, title, abstract, tldr, authors_text, venue)
        SELECT id, title, abstract, tldr, authors_text, venue FROM papers
        """
    )
    conn.commit()


def tokenized_query(query: str) -> List[str]:
    seen = set()
    tokens = []
    for token in TOKEN_PATTERN.findall(query.lower()):
        if len(token) < 2 or token in seen:
            continue
        seen.add(token)
        tokens.append(token)
    return tokens


def build_fts_query(query: str) -> str:
    tokens = tokenized_query(query)
    if not tokens:
        return ""
    return " OR ".join(f'"{token}"*' for token in tokens)


class LocalSemanticScholarSearcher:
    def __init__(
        self,
        db_path: Optional[Union[str, Path]] = None,
        datasets_root: Optional[Union[str, Path]] = None,
    ):
        self.db_path = get_s2_db_path(db_path, datasets_root)

    def _ensure_db_exists(self) -> None:
        if self.db_path.exists():
            return

        datasets_root = get_s2_datasets_root()
        raise FileNotFoundError(
            "Local Semantic Scholar database not found. "
            f"Expected: {self.db_path}. "
            "Build it after downloading datasets with:\n"
            f"  python build_s2_local_db.py --datasets-root {datasets_root}\n"
            "Or switch to the online API backend with:\n"
            "  export S2_SEARCH_BACKEND=api"
        )

    def search(self, query: str, result_limit: int) -> Optional[List[Dict]]:
        if not query or not query.strip():
            return None

        self._ensure_db_exists()
        fts_query = build_fts_query(query)
        if not fts_query:
            return None

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT
                    p.id,
                    p.corpus_id,
                    p.paper_id,
                    p.title,
                    p.authors_json,
                    p.venue,
                    p.year,
                    p.citation_count,
                    p.abstract,
                    p.tldr,
                    p.doi,
                    p.arxiv_id,
                    bm25(papers_fts, 10.0, 4.0, 2.0, 1.5, 1.0) AS score
                FROM papers_fts
                JOIN papers p ON p.id = papers_fts.rowid
                WHERE papers_fts MATCH ?
                ORDER BY score ASC, p.citation_count DESC, p.year DESC
                LIMIT ?
                """,
                (fts_query, result_limit),
            ).fetchall()

            if not rows:
                like_query = f"%{query.lower()}%"
                rows = conn.execute(
                    """
                    SELECT
                        id,
                        corpus_id,
                        paper_id,
                        title,
                        authors_json,
                        venue,
                        year,
                        citation_count,
                        abstract,
                        tldr,
                        doi,
                        arxiv_id
                    FROM papers
                    WHERE lower(title) LIKE ?
                       OR lower(abstract) LIKE ?
                       OR lower(tldr) LIKE ?
                       OR lower(authors_text) LIKE ?
                    ORDER BY citation_count DESC, year DESC
                    LIMIT ?
                    """,
                    (like_query, like_query, like_query, like_query, result_limit),
                ).fetchall()

        if not rows:
            return None

        papers = []
        for row in rows:
            authors = json.loads(row["authors_json"]) if row["authors_json"] else []
            abstract = row["abstract"] or row["tldr"] or ""
            paper = {
                "paperId": row["paper_id"],
                "corpusId": row["corpus_id"],
                "title": row["title"],
                "authors": authors,
                "venue": row["venue"],
                "year": row["year"],
                "abstract": abstract,
                "citationCount": row["citation_count"] or 0,
                "doi": row["doi"] or "",
                "arxivId": row["arxiv_id"] or "",
            }
            paper["citationStyles"] = {"bibtex": _build_bibtex(paper)}
            papers.append(paper)

        return papers


@lru_cache(maxsize=1)
def _get_local_searcher() -> LocalSemanticScholarSearcher:
    return LocalSemanticScholarSearcher()


def _local_db_exists() -> bool:
    return get_s2_db_path().exists()


def _search_for_papers_impl(
    query: str,
    result_limit: int,
    fields: str,
    backend: Optional[str] = None,
) -> Optional[List[Dict]]:
    selected_backend = get_s2_search_backend(backend)
    if selected_backend == "local":
        return _get_local_searcher().search(query, result_limit)

    if selected_backend == "api":
        return _search_for_papers_via_api(query, result_limit, fields)

    if _local_db_exists():
        return _get_local_searcher().search(query, result_limit)

    warnings.warn(
        "Local Semantic Scholar DB was not found; falling back to the online API. "
        "Set S2_SEARCH_BACKEND=local to require the local database."
    )
    return _search_for_papers_via_api(query, result_limit, fields)


class SemanticScholarSearchTool(BaseTool):
    def __init__(
        self,
        name: str = "SearchSemanticScholar",
        description: str = (
            "Search Semantic Scholar literature using either the local downloaded database "
            "or the online API, depending on S2_SEARCH_BACKEND."
        ),
        max_results: int = 10,
    ):
        parameters = [
            {
                "name": "query",
                "type": "str",
                "description": "The search query to find relevant papers.",
            }
        ]
        super().__init__(name, description, parameters)
        self.max_results = max_results
        self.searcher = _get_local_searcher()

    def use_tool(self, query: str) -> Optional[str]:
        papers = self.search_for_papers(query)
        if papers:
            return self.format_papers(papers)
        return "No papers found."

    @backoff.on_exception(
        backoff.expo,
        (requests.exceptions.HTTPError, requests.exceptions.ConnectionError),
        on_backoff=on_backoff,
    )
    def search_for_papers(self, query: str) -> Optional[List[Dict]]:
        selected_backend = get_s2_search_backend()
        papers = _search_for_papers_impl(
            query,
            self.max_results,
            DEFAULT_TOOL_FIELDS,
        )
        total = len(papers) if papers else 0
        if total == 0:
            return None

        papers.sort(key=lambda x: x.get("citationCount", 0), reverse=True)
        if selected_backend == "api" or (
            selected_backend == "auto" and not _local_db_exists()
        ):
            time.sleep(1.0)
        return papers

    def format_papers(self, papers: List[Dict]) -> str:
        paper_strings = []
        for i, paper in enumerate(papers):
            authors = ", ".join(
                author.get("name", "Unknown") for author in paper.get("authors", [])
            )
            paper_strings.append(
                f"""{i + 1}: {paper.get("title", "Unknown Title")}. {authors}. {paper.get("venue", "Unknown Venue")}, {paper.get("year", "Unknown Year")}.
Number of citations: {paper.get("citationCount", "N/A")}
Abstract: {paper.get("abstract", "No abstract available.")}"""
            )
        return "\n\n".join(paper_strings)

@backoff.on_exception(
    backoff.expo,
    (requests.exceptions.HTTPError, requests.exceptions.ConnectionError),
    on_backoff=on_backoff,
)
def search_for_papers(
    query,
    result_limit=10,
    backend: Optional[str] = None,
) -> Union[None, List[Dict]]:
    selected_backend = get_s2_search_backend(backend)
    papers = _search_for_papers_impl(
        query,
        result_limit,
        DEFAULT_WRITEUP_FIELDS,
        backend=backend,
    )
    if not papers:
        return None
    papers.sort(key=lambda x: x.get("citationCount", 0), reverse=True)
    if selected_backend == "api" or (
        selected_backend == "auto" and not _local_db_exists()
    ):
        time.sleep(1.0)
    return papers
