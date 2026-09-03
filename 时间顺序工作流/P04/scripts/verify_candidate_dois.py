"""从集中证据矩阵抽取候选DOI并通过Crossref核验题录存在性。"""

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


ROW = re.compile(r"^\| (L\d{3}) \| (.+?) \|", re.MULTILINE)
DOI = re.compile(r"DOI:\s*([^ |]+)", re.IGNORECASE)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("matrix", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    text = args.matrix.read_text(encoding="utf-8")
    raw_records = ROW.findall(text)

    def verify(raw_record):
        literature_id, citation = raw_record
        match = DOI.search(citation)
        if not match:
            return {"id": literature_id, "doi": None, "status": "no-doi-in-record"}
        doi = match.group(1).rstrip(".,;)")
        url = "https://api.crossref.org/works/" + urllib.parse.quote(doi, safe="")
        request = urllib.request.Request(url, headers={"User-Agent": "P04-literature-audit/1.0 (mailto:research@example.invalid)"})
        try:
            with urllib.request.urlopen(request, timeout=20) as response:
                message = json.load(response)["message"]
            title = (message.get("title") or [""])[0]
            return {"id": literature_id, "doi": doi, "status": "verified", "crossref_title": title}
        except urllib.error.HTTPError as error:
            return {"id": literature_id, "doi": doi, "status": f"http-{error.code}"}
        except Exception as error:
            return {"id": literature_id, "doi": doi, "status": "error", "error": type(error).__name__}

    with ThreadPoolExecutor(max_workers=8) as executor:
        records = list(executor.map(verify, raw_records))
    summary = {
        "candidate_records": len(records),
        "doi_records": sum(record["doi"] is not None for record in records),
        "verified": sum(record["status"] == "verified" for record in records),
        "unverified_doi": [record for record in records if record["doi"] and record["status"] != "verified"],
        "without_doi": [record["id"] for record in records if record["doi"] is None],
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: summary[key] for key in ("candidate_records", "doi_records", "verified", "unverified_doi", "without_doi")}, ensure_ascii=False, indent=2))
    return 0 if not summary["unverified_doi"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
