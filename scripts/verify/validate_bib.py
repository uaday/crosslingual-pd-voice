"""Validate every entry in thesis/references.bib against the registry that issued it.

For each entry:
  * DOI present   -> Crossref (DataCite as fallback): the DOI must resolve, and title,
                     first author's surname, year, volume and first page are compared
  * arXiv id      -> arXiv API: the id must exist and the title must match
  * URL only      -> the URL must answer HTTP 2xx/3xx
  * none of these -> Crossref bibliographic search on title + first author; reported as
                     MATCH only if the top hit's title is near-identical

Titles are compared after removing LaTeX, braces, accents and punctuation (similarity
>= 0.90 counts as a match). Year may differ by one (online-first vs issue year), and that
is reported as a NOTE rather than a failure.

Run:  python scripts/verify/validate_bib.py   (writes thesis/BIB_VALIDATION.md)
"""

from __future__ import annotations

import difflib
import re
import sys
import time
import unicodedata
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[2]
BIB = ROOT / "thesis" / "references.bib"
OUT = ROOT / "thesis" / "BIB_VALIDATION.md"
UA = {"User-Agent": "pdvoice-thesis-bib-check (mailto:none)"}


# ---------------------------------------------------------------- parsing
def parse_bib(text: str) -> list[dict]:
    entries = []
    for m in re.finditer(r"@(\w+)\s*\{\s*([^,\s]+)\s*,", text):
        etype, key = m.group(1).lower(), m.group(2)
        i, depth = m.end(), 1
        while depth and i < len(text):
            depth += {"{": 1, "}": -1}.get(text[i], 0)
            i += 1
        body = text[m.end(): i - 1]
        fields, j = {}, 0
        for fm in re.finditer(r"(\w+)\s*=\s*", body):
            if fm.start() < j:
                continue
            name, k = fm.group(1).lower(), fm.end()
            if body[k] == "{":
                d, s = 1, k + 1
                k += 1
                while d:
                    d += {"{": 1, "}": -1}.get(body[k], 0)
                    k += 1
                val = body[s: k - 1]
            else:
                e = body.find(",", k)
                val = body[k: e if e > 0 else len(body)]
            fields[name] = " ".join(val.split())
            j = k
        entries.append(dict(type=etype, key=key, **fields))
    return entries


ACCENTS = {r"\v": "", r"\'": "", r"\`": "", r'\"': "", r"\^": "", r"\~": "", r"\c": "",
           r"\u": "", r"\=": "", r"\.": ""}


def norm(s: str) -> str:
    s = s or ""
    s = re.sub(r"\\[a-zA-Z]+\s*", " ", s)            # \emph, \url ...
    for a in ACCENTS:
        s = s.replace(a, "")
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"<[^>]+>", " ", s)                     # jats tags in Crossref titles
    s = re.sub(r"[^a-z0-9 ]", " ", s.lower())
    return " ".join(s.split())


def sim(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, norm(a), norm(b)).ratio()


def first_surname(author_field: str) -> str:
    first = re.split(r"\s+and\s+", author_field or "")[0]
    first = first.strip("{} ")
    sur = first.split(",")[0] if "," in first else first.split()[-1] if first.split() else ""
    return norm(sur).split()[-1] if norm(sur) else ""


def get(url: str, **kw):
    for attempt in range(4):
        try:
            r = requests.get(url, headers=UA, timeout=30, **kw)
            if r.status_code == 429:
                time.sleep(3 * (attempt + 1))
                continue
            return r
        except requests.RequestException:
            time.sleep(2 * (attempt + 1))
    return None


# ---------------------------------------------------------------- registries
def _years(m: dict) -> set[int]:
    """Every year Crossref associates with a work: online, print, issued, issue."""
    ys = set()
    for k in ("issued", "published-print", "published-online", "published", "journal-issue"):
        node = m.get(k) or {}
        if k == "journal-issue":
            node = node.get("published-print") or {}
        parts = node.get("date-parts") or [[None]]
        if parts and parts[0] and parts[0][0]:
            ys.add(int(parts[0][0]))
    return ys


def _full_title(m: dict) -> str:
    t = (m.get("title") or [""])[0]
    sub = (m.get("subtitle") or [""])
    return f"{t}: {sub[0]}" if sub and sub[0] and norm(sub[0]) not in norm(t) else t


def crossref(doi: str) -> dict | None:
    r = get(f"https://api.crossref.org/works/{requests.utils.quote(doi, safe='')}")
    if r is None or r.status_code != 200:
        return None
    m = r.json()["message"]
    return dict(source="Crossref", title=_full_title(m), short_title=(m.get("title") or [""])[0],
                surname=norm((m.get("author") or [{}])[0].get("family", "")),
                year=min(_years(m)) if _years(m) else None, years=_years(m),
                volume=m.get("volume"),
                page=(m.get("page") or m.get("article-number") or ""))


def datacite(doi: str) -> dict | None:
    r = get(f"https://api.datacite.org/dois/{requests.utils.quote(doi, safe='')}")
    if r is None or r.status_code != 200:
        return None
    a = r.json()["data"]["attributes"]
    creators = a.get("creators") or [{}]
    return dict(source="DataCite", title=(a.get("titles") or [{}])[0].get("title", ""),
                surname=norm(creators[0].get("familyName") or creators[0].get("name", "")),
                year=a.get("publicationYear"), volume=None, page="")


def arxiv(aid: str) -> dict | None:
    r = get(f"http://export.arxiv.org/api/query?id_list={aid}")
    if r is None or r.status_code != 200 or "<entry>" not in r.text:
        return None
    entry = r.text.split("<entry>", 1)[1]
    title = re.search(r"<title>(.*?)</title>", entry, re.S)
    name = re.search(r"<name>(.*?)</name>", entry, re.S)
    year = re.search(r"<published>(\d{4})", entry)
    if not title or "Error" in title.group(1):
        return None
    return dict(source="arXiv", title=" ".join(title.group(1).split()),
                surname=norm(name.group(1).split()[-1]) if name else "",
                year=int(year.group(1)) if year else None, volume=None, page="")


def crossref_search(title: str, author: str) -> dict | None:
    r = get("https://api.crossref.org/works",
            params={"query.bibliographic": f"{norm(title)} {author}", "rows": 1})
    if r is None or r.status_code != 200 or not r.json()["message"]["items"]:
        return None
    m = r.json()["message"]["items"][0]
    return dict(source="Crossref search", title=_full_title(m),
                short_title=(m.get("title") or [""])[0],
                surname=norm((m.get("author") or [{}])[0].get("family", "")),
                year=min(_years(m)) if _years(m) else None, years=_years(m),
                volume=m.get("volume"),
                page=(m.get("page") or m.get("article-number") or ""), doi=m.get("DOI"))


def dblp(title: str) -> dict | None:
    """DBLP covers NeurIPS, ICML/PMLR, ICLR, JMLR, AAAI and Interspeech, which mostly lack DOIs."""
    data = None
    for attempt in range(4):                           # DBLP answers HTML when throttled
        r = get("https://dblp.org/search/publ/api",
                params={"q": norm(title), "format": "json", "h": 5})
        try:
            data = r.json() if r is not None and r.status_code == 200 else None
        except ValueError:
            data = None
        if data is not None:
            break
        time.sleep(5 * (attempt + 1))
    if data is None:
        return None
    time.sleep(1.0)
    hits = (data.get("result", {}).get("hits", {}) or {}).get("hit", [])
    best = None
    for h in hits:
        info = h.get("info", {})
        s = sim(title, info.get("title", "").rstrip("."))
        if best is None or s > best[0]:
            best = (s, info)
    if not best or best[0] < 0.90:
        return None
    info = best[1]
    authors = info.get("authors", {}).get("author", [])
    authors = authors if isinstance(authors, list) else [authors]
    first = authors[0].get("text", "") if authors else ""
    first = re.sub(r"\s+\d{4}$", "", first)            # DBLP disambiguation suffix
    return dict(source=f"DBLP ({info.get('venue', '?')})", title=info.get("title", "").rstrip("."),
                surname=norm(first.split()[-1]) if first else "",
                year=int(info["year"]) if info.get("year") else None,
                years={int(info["year"])} if info.get("year") else set(),
                volume=info.get("volume"), page=info.get("pages", ""))


def openlibrary(title: str, author: str) -> dict | None:
    r = get("https://openlibrary.org/search.json", params={"title": title, "author": author,
                                                         "limit": 5})
    if r is None or r.status_code != 200:
        return None
    try:
        docs = r.json().get("docs", [])
    except ValueError:
        return None
    for d in docs:
        if sim(title, d.get("title", "")) >= 0.90:
            years = set(d.get("publish_year", []) or [])
            return dict(source="Open Library", title=d.get("title", ""),
                        surname=norm((d.get("author_name") or [""])[0].split()[-1]
                                     if d.get("author_name") else ""),
                        year=min(years) if years else None, years=years, volume=None, page="")
    return None


# ---------------------------------------------------------------- comparison
def compare(e: dict, ref: dict) -> tuple[str, list[str]]:
    problems, notes = [], []
    t = max(sim(e.get("title", ""), ref["title"]),
            sim(e.get("title", ""), ref.get("short_title") or ref["title"]))
    if t < 0.90:
        problems.append(f"title differs ({t:.2f}): registry has '{ref['title'][:90]}'")
    sur = first_surname(e.get("author", ""))
    if ref["surname"] and sur and sur not in ref["surname"] and ref["surname"] not in sur:
        problems.append(f"first author '{sur}' vs registry '{ref['surname']}'")
    try:
        ey = int(e.get("year", 0))
        years = set(ref.get("years") or ({int(ref["year"])} if ref.get("year") else set()))
        if years and ey not in years:
            near = min(years, key=lambda y: abs(y - ey))
            (notes if abs(near - ey) == 1 else problems).append(
                f"year {ey} vs registry {sorted(years)}")
    except ValueError:
        problems.append(f"unparseable year '{e.get('year')}'")
    if ref.get("volume") and e.get("volume") and str(e["volume"]) != str(ref["volume"]):
        problems.append(f"volume {e['volume']} vs registry {ref['volume']}")
    if ref.get("page") and e.get("pages"):
        rp = str(ref["page"]).split("-")[0]
        ep = re.split(r"-+", e["pages"])[0]
        if rp and ep and rp != ep:
            problems.append(f"first page {ep} vs registry {rp}")
    status = "FAIL" if problems else ("NOTE" if notes else "OK")
    return status, problems + notes


def validate(e: dict) -> tuple[str, str, list[str]]:
    doi = e.get("doi", "").strip()
    if doi:
        ref = crossref(doi) or datacite(doi)
        if ref is None:
            return "FAIL", "DOI", [f"DOI {doi} does not resolve at Crossref or DataCite"]
        s, msgs = compare(e, ref)
        return s, ref["source"], msgs
    arx = re.search(r"arXiv[:\s]*(\d{4}\.\d{4,5})", " ".join(e.values()), re.I)
    if arx:
        ref = arxiv(arx.group(1))
        if ref is None:
            return "FAIL", "arXiv", [f"arXiv:{arx.group(1)} not found"]
        s, msgs = compare(e, ref)
        return s, "arXiv", msgs
    url = re.search(r"\\url\{([^}]+)\}", e.get("howpublished", "") + e.get("url", ""))
    if url:
        r = get(url.group(1), allow_redirects=True)
        ok = r is not None and r.status_code < 400
        return ("OK" if ok else "FAIL"), "URL", ([] if ok else [f"URL {url.group(1)} -> "
                                                              f"{getattr(r, 'status_code', 'no response')}"])
    title, sur = e.get("title", ""), first_surname(e.get("author", ""))
    if e["type"] == "book":
        ref = openlibrary(title, sur)
        if ref:
            s, msgs = compare(e, ref)
            return s, ref["source"], msgs
        return "NOTE", "book", ["book not found in Open Library; verify by hand"]
    ref = dblp(title)
    if ref:
        s, msgs = compare(e, ref)
        return s, ref["source"], msgs
    ref = crossref_search(title, sur)
    if ref and sim(title, ref["title"]) >= 0.90:
        s, msgs = compare(e, ref)
        hint = [f"no DOI in bib; registry DOI {ref.get('doi')}"] if ref.get("doi") else []
        return ("NOTE" if s == "OK" else s), "Crossref search", hint + msgs
    return "NOTE", "none", ["no DOI/arXiv/URL and no confident registry match; verify by hand"]


def main() -> None:
    entries = parse_bib(BIB.read_text(encoding="utf-8"))
    used = set()
    for f in (ROOT / "thesis").glob("*.tex"):
        for c in re.findall(r"\\cite\{([^}]*)\}", f.read_text(encoding="utf-8")):
            used |= {k.strip() for k in c.split(",")}
    rows = []
    for n, e in enumerate(entries, 1):
        status, src, msgs = validate(e)
        rows.append((status, e["key"], src, "; ".join(msgs), e["key"] in used))
        print(f"[{n:3d}/{len(entries)}] {status:<4} {e['key']:<24} {src:<16} {'; '.join(msgs)[:110]}",
              flush=True)
        time.sleep(0.2)
    keys = [e["key"] for e in entries]
    dupes = sorted({k for k in keys if keys.count(k) > 1})
    missing = sorted(used - set(keys))
    counts = {s: sum(r[0] == s for r in rows) for s in ("OK", "NOTE", "FAIL")}
    lines = [f"# Bibliography validation ({time.strftime('%Y-%m-%d')})", "",
             f"{len(entries)} entries: **{counts['OK']} OK, {counts['NOTE']} NOTE, "
             f"{counts['FAIL']} FAIL**. Cited but missing from the bib: {missing or 'none'}. "
             f"Duplicate keys: {dupes or 'none'}. Uncited entries: "
             f"{sorted(set(keys) - used) or 'none'}.", "",
             "Generated by `scripts/verify/validate_bib.py`; do not edit by hand.", "",
             "| status | key | checked against | detail | cited |", "|---|---|---|---|---|"]
    order = {"FAIL": 0, "NOTE": 1, "OK": 2}
    for r in sorted(rows, key=lambda r: (order[r[0]], r[1])):
        lines.append(f"| {r[0]} | `{r[1]}` | {r[2]} | {r[3].replace('|', '/')} | "
                     f"{'yes' if r[4] else '**no**'} |")
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n{counts}  missing={missing}  dupes={dupes}\nwrote {OUT}")
    sys.exit(1 if counts["FAIL"] or missing or dupes else 0)


if __name__ == "__main__":
    main()
