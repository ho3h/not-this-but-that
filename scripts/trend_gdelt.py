"""Fetch GDELT DOC 2.0 daily counts for "not X but Y" phrases and controls in English-language news.

GDELT asks for at most one request every 5 s, so requests are spaced 12 s apart and each raw
response is cached in data/trend/gdelt/<name>.json (re-runs skip cached queries). Mode
timelinevolraw returns, per day, the number of matching articles ("value") and the number of
articles GDELT monitored that day ("norm").

Phrase rates are later expressed relative to controls fetched the same way, so changes in
GDELT's source mix cancel out:
  - "according to": a neutral news phrase, the volume baseline;
  - "not only": a contrast phrase with no chatbot association, the phrase control.
("isn't" was planned as a contraction control, but GDELT refused it every time, even for a
single year, presumably because the word is too common.)
Press-wire queries repeat the phrase and the volume control inside one domain each.
"""

from __future__ import annotations

import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "data/trend/gdelt"
START, END = "20170101000000", "20261009000000"
LANG = " sourcelang:english"
PHRASES = {"isnt_just": '"isn\'t just"', "according_to": '"according to"', "not_only": '"not only"',
           "not_just": '"not just"', "more_than_just": '"more than just"',
           "is_not_just": '"is not just"', "its_not_just": '"it\'s not just"'}
DOMAINS = {"prn": "prnewswire.com", "bw": "businesswire.com", "gnw": "globenewswire.com"}


def queries() -> dict[str, str]:
    out = {name: q + LANG for name, q in PHRASES.items()}
    for tag, dom in DOMAINS.items():
        for name in ("isnt_just", "according_to"):
            out[f"{name}__{tag}"] = f"{PHRASES[name]} domain:{dom}"
    return out


def fetch_range(q: str, start: str, end: str) -> str:
    url = "https://api.gdeltproject.org/api/v2/doc/doc?" + urllib.parse.urlencode(
        {"query": q, "mode": "timelinevolraw", "format": "json", "startdatetime": start, "enddatetime": end,
         "timelinesmooth": 0})
    req = urllib.request.Request(url, headers={"User-Agent": "not-this-but-that trend study"})
    return urllib.request.urlopen(req, timeout=180).read().decode()


def fetch(q: str) -> str:
    """Whole range in one request; if GDELT refuses, fall back to one request per year, merged."""
    try:
        body = fetch_range(q, START, END)
        if body.lstrip().startswith("{"):
            return body
    except Exception:
        pass
    import hashlib

    days, details = [], None
    chunks = OUT / "chunks"
    chunks.mkdir(exist_ok=True)
    key = hashlib.sha1(q.encode()).hexdigest()[:10]
    for year in range(int(START[:4]), int(END[:4]) + 1):
        lo = f"{year}0101000000" if year > int(START[:4]) else START
        hi = f"{year + 1}0101000000" if year < int(END[:4]) else END
        cached = chunks / f"{key}_{year}_{hi}.json"  # the end stamp changes only for the current year
        if cached.exists():
            d = json.loads(cached.read_text())
            details = details or d["query_details"]
            days += [x for x in d["timeline"][0]["data"] if x["date"][:4] == str(year)] if d.get("timeline") else []
            continue
        for attempt in range(6):
            time.sleep(20 + 40 * attempt)
            try:
                part = fetch_range(q, lo, hi)
            except Exception as e:
                part = f"ERROR {e}"
            if part.lstrip().startswith("{"):
                cached.write_text(part)
                d = json.loads(part)
                details = details or d["query_details"]
                if d.get("timeline"):
                    days += [x for x in d["timeline"][0]["data"] if x["date"][:4] == str(year)]
                break
        else:
            return f"ERROR year {year} refused"
    return json.dumps({"query_details": details, "timeline": [{"series": "Article Count", "data": days}]})


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    first = True
    for name, q in queries().items():
        path = OUT / f"{name}.json"
        if path.exists() and path.read_text().lstrip().startswith("{"):
            continue
        if not first:
            time.sleep(12)
        first = False
        for attempt in range(4):
            try:
                body = fetch(q)
            except Exception as e:  # network hiccup: back off and retry
                body = f"ERROR {e}"
            if body.lstrip().startswith("{"):
                break
            print(f"{name}: not JSON ({body[:80]!r}), waiting", flush=True)
            time.sleep(60 * (attempt + 1))
        path.write_text(body)
        ok = body.lstrip().startswith("{")
        n = len(json.loads(body)["timeline"][0]["data"]) if ok and json.loads(body).get("timeline") else 0
        print(f"{name}: {'ok' if ok else 'FAILED'} ({n} days)", flush=True)
    sys.exit(0)


if __name__ == "__main__":
    main()
