"""
Citation verification tool.

Parses a BibTeX file and a directory of .tex files, then:

1. Extracts every bib entry's DOI / arXiv ID / title and verifies it against
   the CrossRef API (always) and the NASA ADS API (if ``ADS_API_TOKEN`` is
   set), flagging entries whose DOI does not resolve or resolves to a paper
   whose title does not match the bib entry's stated title.
2. Extracts every \\cite / \\citep / \\citet (and friends) command from the
   .tex files, together with the sentence/clause it is attached to, and
   flags any citekey used in the .tex files that has no matching entry in
   the bib file.
3. For every citation where the cited paper's abstract could be fetched
   (arXiv / CrossRef / ADS), pairs the claim sentence with that abstract so
   a downstream reasoning step (an LLM, a human, ...) can judge whether the
   paper plausibly supports the specific claim it is attached to. This
   script does NOT make that judgment itself -- it has no way to reason
   about semantic plausibility. It only gathers the (claim, abstract) pairs.

Output is a single JSON file (default: citation_audit.json) with three
top-level keys: ``bib_issues``, ``missing_citekeys``, and
``semantic_review_candidates``. A short mechanical-only summary (points 1
and 2 above) is also printed to stdout, since those checks are fully
automatic and don't require further reasoning.

This script only uses the standard library -- no extra pip installs needed.

Usage:
    python scripts/verify_citations.py --bib report/references.bib --tex-dir report
    python scripts/verify_citations.py --bib report/references.bib --tex-dir report \\
        --out report/citation_audit.json --sleep 0.2

Network notes:
    - CrossRef's public API needs no key. Set CROSSREF_MAILTO to add a
      contact email to the User-Agent (CrossRef's "polite pool"), which
      gets you more reliable rate limits.
    - ADS (api.adsabs.harvard.edu) requires a bearer token. Set
      ADS_API_TOKEN to enable it; otherwise ADS lookups are skipped and
      CrossRef/arXiv are used alone.
"""

from __future__ import annotations

import argparse
import difflib
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from xml.etree import ElementTree

USER_AGENT = "citation-verification-tool/1.0 (mailto:{})".format(
    os.environ.get("CROSSREF_MAILTO", "no-reply@example.com")
)
ADS_API_TOKEN = os.environ.get("ADS_API_TOKEN", "")

ARXIV_ID_RE = re.compile(r"^\d{4}\.\d{4,5}(v\d+)?$|^[a-zA-Z\-\.]+/\d{7}$")

# LaTeX abbreviations whose trailing "." should not be treated as a sentence
# boundary when extracting the claim context around a citation.
NON_TERMINAL_ABBREVS = (
    "et al",
    "e.g",
    "i.e",
    "cf",
    "vs",
    "etc",
    "Fig",
    "Eq",
    "Eqs",
    "Sec",
    "Sect",
    "Ref",
    "Refs",
    "Tab",
    "No",
    "eq",
    "fig",
    "sec",
    "resp",
    "approx",
    "Dr",
    "Mr",
    "Mrs",
    "Prof",
    "vol",
    "Vol",
    "pp",
    "p",
)

SENTENCE_PLACEHOLDER = "\x00"


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class BibEntry:
    key: str
    entrytype: str
    title: str = ""
    doi: str = ""
    arxiv_id: str = ""


@dataclass
class CitationUse:
    file: str
    line: int
    command: str
    keys: list[str]
    context: str


@dataclass
class VerificationResult:
    key: str
    status: str  # ok | title_mismatch | not_found | no_identifier | lookup_error
    lookup_source: str = ""
    lookup_title: str = ""
    lookup_doi: str = ""
    similarity: float | None = None
    abstract: str = ""
    detail: str = ""


# ---------------------------------------------------------------------------
# BibTeX parsing (regex-based, no third-party dependency)
# ---------------------------------------------------------------------------


def split_bib_entries(text: str) -> list[str]:
    return re.split(r"\n(?=@\w+\{)", text)


def extract_field(entry_text: str, field_name: str) -> str:
    """Extract a bib field's value, tolerating brace- or quote-delimited
    values and one level of nested braces (e.g. accented LaTeX macros)."""
    m = re.search(
        r"(?<![a-zA-Z])" + re.escape(field_name) + r"\s*=\s*\{((?:[^{}]|\{[^{}]*\})*)\}",
        entry_text,
        re.IGNORECASE,
    )
    if m:
        return m.group(1).strip()
    m = re.search(
        r"(?<![a-zA-Z])" + re.escape(field_name) + r'\s*=\s*"([^"]*)"',
        entry_text,
        re.IGNORECASE,
    )
    return m.group(1).strip() if m else ""


def doi_from_url(s: str) -> str:
    m = re.search(r"doi\.org/(10\.\S+)", s)
    return m.group(1).rstrip(",}") if m else ""


def parse_bib(bib_path: Path) -> dict[str, BibEntry]:
    text = bib_path.read_text(encoding="utf-8", errors="replace")
    entries: dict[str, BibEntry] = {}
    for block in split_bib_entries(text):
        m = re.match(r"@(\w+)\{([^,]+),", block)
        if not m:
            continue
        entrytype, key = m.group(1), m.group(2).strip()
        title = extract_field(block, "title")
        doi = extract_field(block, "doi")
        eprint = extract_field(block, "eprint")

        resolved_doi = ""
        if doi.startswith("10."):
            resolved_doi = doi
        elif doi.startswith("http") and doi_from_url(doi):
            resolved_doi = doi_from_url(doi)
        elif key.startswith("10."):
            resolved_doi = key
        elif key.startswith("doi:10."):
            resolved_doi = key[4:]
        elif key.startswith("https://doi.org/10."):
            resolved_doi = doi_from_url(key)

        arxiv_id = ""
        if ARXIV_ID_RE.match(eprint):
            arxiv_id = eprint
        elif not resolved_doi and eprint.startswith("http") and doi_from_url(eprint):
            resolved_doi = doi_from_url(eprint)

        entries[key] = BibEntry(
            key=key, entrytype=entrytype, title=title, doi=resolved_doi, arxiv_id=arxiv_id
        )
    return entries


# ---------------------------------------------------------------------------
# Network lookups
# ---------------------------------------------------------------------------


def http_get(
    url: str, headers: dict | None = None, timeout: float = 20.0, retries: int = 3
) -> bytes | None:
    req = urllib.request.Request(url, headers=headers or {"User-Agent": USER_AGENT})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except urllib.error.HTTPError as exc:
            if exc.code == 429 and attempt < retries - 1:
                wait = float(exc.headers.get("Retry-After", 3.0 * (attempt + 1)))
                time.sleep(wait)
                continue
            return None
        except (urllib.error.URLError, TimeoutError):
            return None
    return None


def crossref_lookup(doi: str) -> dict | None:
    raw = http_get(f"https://api.crossref.org/works/{doi}")
    if raw is None:
        return None
    try:
        data = json.loads(raw)["message"]
    except (json.JSONDecodeError, KeyError):
        return None
    # CrossRef sometimes splits a paper's full title into a short "title"
    # (e.g. "Planck 2018 results") and a separate "subtitle" (e.g. "IV.
    # Diffuse component separation"); join them so comparisons against a
    # bib entry's full title aren't spuriously flagged as mismatches.
    title_parts = (data.get("title") or []) + (data.get("subtitle") or [])
    title = ". ".join(t for t in title_parts if t).strip()
    title = re.sub(r"\s*<[^>]+>\s*", " ", title)
    title = re.sub(r"\s+", " ", title).strip()
    abstract = re.sub(r"\s*<[^>]+>\s*", " ", data.get("abstract", "") or "")
    abstract = re.sub(r"^\s*Abstract\s+", "", abstract, flags=re.IGNORECASE)
    abstract = re.sub(r"\s+", " ", abstract).strip()
    return {"title": title, "doi": data.get("DOI", doi), "abstract": abstract}


_last_arxiv_call = 0.0
ARXIV_MIN_INTERVAL = 3.0  # arXiv's API guidance: no more than ~1 request / 3s


def arxiv_lookup(arxiv_id: str) -> dict | None:
    global _last_arxiv_call
    wait = ARXIV_MIN_INTERVAL - (time.monotonic() - _last_arxiv_call)
    if wait > 0:
        time.sleep(wait)
    _last_arxiv_call = time.monotonic()
    raw = http_get(f"https://export.arxiv.org/api/query?id_list={arxiv_id}")
    if raw is None:
        return None
    try:
        root = ElementTree.fromstring(raw)
    except ElementTree.ParseError:
        return None
    ns = {"a": "http://www.w3.org/2005/Atom"}
    entry = root.find("a:entry", ns)
    if entry is None:
        return None
    title_el = entry.find("a:title", ns)
    summary_el = entry.find("a:summary", ns)
    title = re.sub(r"\s+", " ", (title_el.text or "").strip()) if title_el is not None else ""
    abstract = (
        re.sub(r"\s+", " ", (summary_el.text or "").strip()) if summary_el is not None else ""
    )
    return {"title": title, "doi": f"10.48550/arXiv.{arxiv_id}", "abstract": abstract}


def ads_lookup(doi: str = "", arxiv_id: str = "") -> dict | None:
    if not ADS_API_TOKEN:
        return None
    q = f"doi:{doi}" if doi else f"arXiv:{arxiv_id}"
    url = (
        "https://api.adsabs.harvard.edu/v1/search/query?"
        f"q={urllib.parse.quote(q)}&fl=title,abstract,doi"
    )
    raw = http_get(
        url, headers={"Authorization": f"Bearer {ADS_API_TOKEN}", "User-Agent": USER_AGENT}
    )
    if raw is None:
        return None
    try:
        docs = json.loads(raw)["response"]["docs"]
    except (json.JSONDecodeError, KeyError, IndexError):
        return None
    if not docs:
        return None
    doc = docs[0]
    title = (doc.get("title") or [""])[0]
    abstract = doc.get("abstract") or ""
    doi_out = (
        (doc.get("doi") or [doi])[0] if isinstance(doc.get("doi"), list) else doc.get("doi", doi)
    )
    return {"title": title, "doi": doi_out, "abstract": abstract}


def normalize_title(t: str) -> str:
    t = re.sub(r"<[^>]+>", "", t)
    t = re.sub(r"[\{\}\\'\"]", "", t)
    t = re.sub(r"\s+", " ", t).strip().lower()
    return t


def verify_entry(entry: BibEntry, sleep: float) -> VerificationResult:
    if not entry.doi and not entry.arxiv_id:
        return VerificationResult(
            key=entry.key, status="no_identifier", detail="No DOI or arXiv ID found in this entry."
        )

    result = None
    source = ""

    # Prefer ADS for astro-style entries when a token is configured, since
    # it typically has the most complete metadata for astronomy papers.
    if ADS_API_TOKEN:
        result = ads_lookup(doi=entry.doi, arxiv_id=entry.arxiv_id)
        source = "ads"
        time.sleep(sleep)

    if result is None and entry.doi:
        result = crossref_lookup(entry.doi)
        source = "crossref"
        time.sleep(sleep)

    if (result is None or not result.get("abstract")) and entry.arxiv_id:
        arxiv_result = arxiv_lookup(entry.arxiv_id)
        time.sleep(sleep)
        if result is None:
            result = arxiv_result
            source = "arxiv"
        elif arxiv_result and not result.get("abstract"):
            result["abstract"] = arxiv_result.get("abstract", "")
            source += "+arxiv_abstract"

    if result is None:
        return VerificationResult(
            key=entry.key,
            status="not_found",
            lookup_source=source,
            detail=f"DOI/arXiv ID did not resolve (doi={entry.doi!r}, arxiv={entry.arxiv_id!r}).",
        )

    similarity = None
    status = "ok"
    detail = ""
    if entry.title and result.get("title"):
        similarity = difflib.SequenceMatcher(
            None, normalize_title(entry.title), normalize_title(result["title"])
        ).ratio()
        if similarity < 0.85:
            status = "title_mismatch"
            detail = "Bib title and looked-up title diverge; this may resolve to a different paper."
    elif not entry.title:
        detail = "Bib entry has no title field to compare against."

    return VerificationResult(
        key=entry.key,
        status=status,
        lookup_source=source,
        lookup_title=result.get("title", ""),
        lookup_doi=result.get("doi", ""),
        similarity=similarity,
        abstract=result.get("abstract", ""),
        detail=detail,
    )


# ---------------------------------------------------------------------------
# .tex parsing
# ---------------------------------------------------------------------------

CITE_RE = re.compile(r"\\([a-zA-Z]*[Cc]ite[a-zA-Z]*)\*?(?:\[[^\]]*\])*\{([^}]+)\}")


def strip_tex_comments(line: str) -> str:
    # Remove unescaped % comments (a naive but adequate heuristic for
    # typical report .tex files).
    out = []
    escaped = False
    for ch in line:
        if ch == "%" and not escaped:
            break
        escaped = (ch == "\\") and not escaped
        out.append(ch)
    return "".join(out)


def split_sentences(paragraph: str) -> list[str]:
    protected = paragraph
    for abbr in NON_TERMINAL_ABBREVS:
        protected = re.sub(re.escape(abbr) + r"\.", abbr + SENTENCE_PLACEHOLDER, protected)
    protected = re.sub(r"(\d)\.(\d)", r"\1" + SENTENCE_PLACEHOLDER + r"\2", protected)
    protected = re.sub(r"\b([A-Z])\.", r"\1" + SENTENCE_PLACEHOLDER, protected)

    pieces = re.split(r"(?<=[.!?])\s+(?=[A-Z\\])", protected)
    return [p.replace(SENTENCE_PLACEHOLDER, ".").strip() for p in pieces if p.strip()]


def extract_citations(tex_path: Path, root: Path) -> list[CitationUse]:
    text = tex_path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    cleaned_lines = [strip_tex_comments(line) for line in lines]

    # Group into paragraphs (blank-line separated) so sentence splitting
    # can see full context even when a citation sits mid-paragraph across
    # a line wrap. Each paragraph also records, for every constituent line,
    # the character offset at which that line starts within the joined
    # para_text -- this lets us recover an exact line number for any match.
    paragraphs: list[tuple[str, list[tuple[int, int]]]] = []  # (text, [(char_offset, line_no)])
    buf: list[str] = []
    buf_line_nos: list[int] = []
    for i, line in enumerate(cleaned_lines):
        if line.strip() == "":
            if buf:
                offsets = []
                pos = 0
                for ln, seg in zip(buf_line_nos, buf):
                    offsets.append((pos, ln))
                    pos += len(seg) + 1  # +1 for the " " joiner
                paragraphs.append((" ".join(buf), offsets))
                buf, buf_line_nos = [], []
        else:
            buf.append(line)
            buf_line_nos.append(i)
    if buf:
        offsets = []
        pos = 0
        for ln, seg in zip(buf_line_nos, buf):
            offsets.append((pos, ln))
            pos += len(seg) + 1
        paragraphs.append((" ".join(buf), offsets))

    def line_for_offset(offsets: list[tuple[int, int]], char_pos: int) -> int:
        line_no = offsets[0][1]
        for off, ln in offsets:
            if off <= char_pos:
                line_no = ln
            else:
                break
        return line_no + 1  # 1-indexed

    citations: list[CitationUse] = []
    for para_text, line_offsets in paragraphs:
        sentences = split_sentences(para_text)
        # Map each citation match (by character offset in para_text) to the
        # sentence containing it.
        offset = 0
        sentence_spans = []
        for s in sentences:
            idx = para_text.find(s, offset)
            if idx == -1:
                idx = offset
            sentence_spans.append((idx, idx + len(s), s))
            offset = idx + len(s)

        for m in CITE_RE.finditer(para_text):
            command = m.group(1)
            keys = [k.strip() for k in m.group(2).split(",")]
            pos = m.start()
            context = para_text[max(0, pos - 200) : pos + 200].strip()
            for s_start, s_end, s_text in sentence_spans:
                if s_start <= pos < s_end + 2:
                    context = s_text
                    break
            citations.append(
                CitationUse(
                    file=str(tex_path.relative_to(root)),
                    line=line_for_offset(line_offsets, pos),
                    command=command,
                    keys=keys,
                    context=context,
                )
            )
    return citations


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--bib", required=True, type=Path, help="Path to references.bib")
    ap.add_argument("--tex-dir", required=True, type=Path, help="Directory to scan for .tex files")
    ap.add_argument(
        "--out", type=Path, default=Path("citation_audit.json"), help="Output JSON path"
    )
    ap.add_argument("--sleep", type=float, default=0.2, help="Delay between API calls (seconds)")
    ap.add_argument(
        "--limit", type=int, default=0, help="Only verify the first N bib entries (0 = all)"
    )
    args = ap.parse_args()

    root = args.tex_dir.resolve()

    print(f"Parsing {args.bib} ...")
    bib_entries = parse_bib(args.bib)
    print(f"  {len(bib_entries)} bib entries found.")

    tex_files = sorted(p.resolve() for p in args.tex_dir.rglob("*.tex"))
    print(f"Scanning {len(tex_files)} .tex file(s) under {args.tex_dir} ...")
    all_citations: list[CitationUse] = []
    for tf in tex_files:
        all_citations.extend(extract_citations(tf, root))
    print(f"  {len(all_citations)} citation commands found.")

    missing_key_locations: dict[str, list[str]] = {}
    for c in all_citations:
        for k in c.keys:
            if k not in bib_entries:
                missing_key_locations.setdefault(k, []).append(f"{c.file}:{c.line}")
    missing_citekeys = [
        {"citekey": k, "used_at": sorted(set(locs))}
        for k, locs in sorted(missing_key_locations.items())
    ]

    print(
        f"Verifying {len(bib_entries)} bib entries against CrossRef"
        + (" + ADS" if ADS_API_TOKEN else "")
        + " ..."
    )
    if not ADS_API_TOKEN:
        print("  (ADS_API_TOKEN not set -- using CrossRef/arXiv only.)")

    verifications: dict[str, VerificationResult] = {}
    items = list(bib_entries.items())
    if args.limit:
        items = items[: args.limit]
    for i, (key, entry) in enumerate(items, 1):
        try:
            verifications[key] = verify_entry(entry, args.sleep)
        except Exception as exc:  # noqa: BLE001 -- keep going on any single-entry failure
            verifications[key] = VerificationResult(key=key, status="lookup_error", detail=str(exc))
        if i % 10 == 0 or i == len(items):
            print(f"  {i}/{len(items)} verified")

    bib_issues = [asdict(v) for v in verifications.values() if v.status not in ("ok",)]

    semantic_review_candidates = []
    for c in all_citations:
        for key in c.keys:
            v = verifications.get(key)
            if v and v.abstract:
                semantic_review_candidates.append(
                    {
                        "file": c.file,
                        "line": c.line,
                        "command": c.command,
                        "citekey": key,
                        "claim_context": c.context,
                        "paper_title": v.lookup_title,
                        "paper_abstract": v.abstract,
                    }
                )

    output = {
        "bib_path": str(args.bib),
        "tex_dir": str(args.tex_dir),
        "n_bib_entries": len(bib_entries),
        "n_citation_commands": len(all_citations),
        "bib_issues": bib_issues,
        "missing_citekeys": missing_citekeys,
        "semantic_review_candidates": semantic_review_candidates,
    }
    args.out.write_text(json.dumps(output, indent=2), encoding="utf-8")

    print()
    print("=" * 70)
    print(f"Bib issues (broken/mismatched/no-identifier): {len(bib_issues)}")
    for issue in bib_issues:
        print(f"  [{issue['status']}] {issue['key']}: {issue['detail']}")
    print(f"Missing citekeys (used in .tex, absent from bib): {len(missing_citekeys)}")
    for m in missing_citekeys:
        print(f"  {m['citekey']}  (used at: {', '.join(m['used_at'])})")
    print(f"Semantic review candidates written to {args.out}: {len(semantic_review_candidates)}")
    print("=" * 70)
    print(f"\nFull structured output written to {args.out}")
    print(
        "Run the semantic plausibility pass (claim vs. abstract) over "
        "'semantic_review_candidates' to produce the final report -- "
        "this script does not judge semantic plausibility itself."
    )


if __name__ == "__main__":
    main()
