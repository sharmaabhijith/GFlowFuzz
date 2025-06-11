#!/usr/bin/env python3
"""
collect_pytorch_bugs.py (verbose edition)

Same functionality as before but with detailed, timestamped progress logs
and periodic file flushing for safer, observable scraping.
"""

from __future__ import annotations
import argparse, json, os, re, sys, time
from datetime import date, datetime, timedelta
from typing import Iterable, Iterator, Dict, List

import requests
from requests.adapters import HTTPAdapter, Retry

# ── Regexes ──────────────────────────────────────────────────────────────────
RE_CODE_BLOCK = re.compile(r"```(?:python)?\s*([\s\S]*?)```", re.I)
RE_TORCH_API  = re.compile(r"\b(torch\.[A-Za-z0-9_\.]+)", re.M)
RE_ERR_LINE   = re.compile(r"^\s*(Traceback|[A-Za-z_]+Error|Exception):", re.M)

# ── Bug-like keywords ─────────────────────────────────────────────────────────
BUG_KEYWORDS = [
    "fix", "defect", "error", "bug", "issue",
    "mistake", "correct", "fault"
]
KEYWORD_CLAUSE = " OR ".join(f"{kw} in:title" for kw in BUG_KEYWORDS)

# ── GitHub API setup ─────────────────────────────────────────────────────────
HEADERS = {
    "Accept": "application/vnd.github+json",
    "Authorization": f"Bearer {os.getenv('GITHUB_TOKEN', '')}".strip()
}
BASE   = "https://api.github.com"
SEARCH = f"{BASE}/search/issues"

# ── Logging helper ───────────────────────────────────────────────────────────
def log(msg: str, *, end: str = "\n") -> None:
    ts = datetime.utcnow().isoformat(" ", "seconds")
    print(f"[{ts} UTC] {msg}", file=sys.stderr, end=end, flush=True)

# ── Utils ────────────────────────────────────────────────────────────────────
def daterange(start: date, end: date, step: int = 10) -> Iterator[tuple[date, date]]:
    """Yield consecutive inclusive 10-day windows."""
    current, delta = start, timedelta(days=step)
    while current < end:
        nxt = min(current + delta, end)
        yield current, nxt - timedelta(days=1)
        current = nxt

def backoff(resp: requests.Response) -> None:
    reset = int(resp.headers.get("X-RateLimit-Reset", time.time() + 60))
    wait  = max(reset - time.time() + 1, 1)
    log(f"Rate-limited → sleeping {wait:.0f}s")
    time.sleep(wait)

def gh_session() -> requests.Session:
    s = requests.Session()
    retries = Retry(total=5, backoff_factor=2,
                    status_forcelist=(500, 502, 503, 504, 403))
    s.mount("https://", HTTPAdapter(max_retries=retries))
    return s

# ── helper: paginate ONE query string & tag its results ─────────────────────
def _paginate(session: requests.Session, q: str, tag: str) -> Iterable[tuple[dict, str]]:
    page = 1
    while True:
        params = {"q": q, "per_page": 100, "page": page}
        resp   = session.get(SEARCH, params=params, headers=HEADERS)
        if resp.status_code == 403:
            backoff(resp); continue
        resp.raise_for_status()
        items = resp.json()["items"]
        log(f"{tag} page {page}: {len(items):3d} results")
        for it in items:
            yield it, tag
        if "next" not in resp.links:
            break
        page += 1
        time.sleep(0.25)          # polite pacing


# ── search_items: same signature but now yields (item, tag) ─────────────────
def search_items(session: requests.Session,
                 artefact: str,
                 start: date,
                 end: date) -> Iterable[tuple[dict, str]]:
    date_rng = f"created:{start.isoformat()}..{end.isoformat()} "
    tail     = "is:issue is:closed linked:pr" if artefact == "issue" else "is:pr"

    # label:bug
    label_q = f"repo:pytorch/pytorch label:bug {date_rng}{tail}"
    yield from _paginate(session, label_q, "label")

    # one query per keyword
    for kw in BUG_KEYWORDS:
        kw_q = f"repo:pytorch/pytorch {kw} in:title {date_rng}{tail}"
        yield from _paginate(session, kw_q, kw)


# ── Snippet extraction ───────────────────────────────────────────────────────
def clean_snippet(code: str) -> str | None:
    if RE_ERR_LINE.search(code):
        code = RE_ERR_LINE.sub("", code)
    code = code.strip()
    if not code or code.count("\n") > 400:
        return None
    return code

def extract_examples(item: dict) -> List[Dict[str, str]]:
    body = item.get("body") or ""
    rows: List[Dict[str, str]] = []
    for snip in RE_CODE_BLOCK.findall(body):
        code = clean_snippet(snip)
        if not code:
            continue
        m = RE_TORCH_API.search(code)
        if not m:
            continue
        rows.append({
            "API": m.group(1),
            "Bug Description": item["title"].strip(),
            "Code": code
        })
    return rows

# ── collect: open one file-handle per tag ───────────────────────────────────
def collect(start_year: int,
            out_dir: str,
            flush_every: int = 250) -> None:

    os.makedirs(out_dir, exist_ok=True)
    file_paths = {tag: os.path.join(out_dir, f"Tag_{tag}.jsonl")
                  for tag in BUG_KEYWORDS + ["label"]}
    files      = {tag: open(fp, "w", encoding="utf-8")
                  for tag, fp in file_paths.items()}

    session, seen_ids = gh_session(), set()
    total_items = total_rows = 0; rows_since_flush = 0

    try:
        for w_start, w_end in daterange(date(start_year, 1, 1), date.today()):
            log(f"▶ window {w_start} → {w_end} START")
            for artefact in ("issue", "pr"):
                for item, tag in search_items(session, artefact, w_start, w_end):
                    if item["id"] in seen_ids:
                        continue
                    seen_ids.add(item["id"])
                    total_items += 1
                    for row in extract_examples(item):
                        files[tag].write(json.dumps(row, ensure_ascii=False)+"\n")
                        total_rows += 1; rows_since_flush += 1
                        if rows_since_flush >= flush_every:
                            for fh in files.values():
                                fh.flush(); os.fsync(fh.fileno())
                            log(f"  ⤷ flushed ({total_rows} rows so far)")
                            rows_since_flush = 0
            log(f"✓ window {w_start} → {w_end} DONE "
                f"(cum items {total_items}, rows {total_rows})")
    finally:                       # ensure clean close
        for fh in files.values():
            fh.flush(); os.fsync(fh.fileno()); fh.close()

    log("Finished. Written files:")
    for tag, fp in file_paths.items():
        log(f"  {tag:>8} → {fp}")


# ── CLI ──────────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--output", default="datasets",
                    help="Path to JSONL output file")
    ap.add_argument("--start-year", type=int, default=2016,
                    help="Earliest year to fetch (default: 2016)")
    args = ap.parse_args()

    if not HEADERS["Authorization"]:
        log("WARNING: no GITHUB_TOKEN set – low rate limits!")

    collect(args.start_year, args.output)

if __name__ == "__main__":
    main()