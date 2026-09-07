"""条件动作景观独立审计。"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


EXPECTED_ACTIONS = {
    (fraction, neighborhood, budget, repeat)
    for fraction in (0.25, 0.5)
    for neighborhood in range(1, 7)
    for budget in (4, 12)
    for repeat in range(4)
}


def _key(row: dict) -> tuple:
    return (
        row["state"],
        float(row["fraction"]),
        int(row["n"]),
        int(row["b"]),
        int(row["repeat"]),
    )


def _candidate_signature(candidate: dict) -> str:
    return json.dumps(
        {
            "chromosome": candidate["chromosome"],
            "objective": candidate["objective"],
            "decode_index": candidate["decode_index"],
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def audit_records(
    manifest: dict, rows: list[dict], raw_rows: list[dict], current_hashes: dict
) -> dict:
    keys = [_key(row) for row in rows]
    counts = Counter(keys)
    duplicate_pairs = sum(count - 1 for count in counts.values() if count > 1)
    states = sorted({row["state"] for row in rows})
    missing_pairs = 0
    for state in states:
        observed = {key[1:] for key in keys if key[0] == state}
        missing_pairs += len(EXPECTED_ACTIONS - observed)
    over_budget = sum(int(row["decodes"]) > int(row["b"]) for row in rows)
    raw_by_key = {_key(row): row for row in raw_rows}
    prefix_mismatches = 0
    prefixes = defaultdict(dict)
    for row in raw_rows:
        prefixes[(row["state"], float(row["fraction"]), int(row["n"]), int(row["repeat"]))][
            int(row["b"])
        ] = row
    for pair in prefixes.values():
        if set(pair) != {4, 12}:
            prefix_mismatches += 1
            continue
        cutoff = int(pair[4]["decodes"])
        short = [_candidate_signature(item) for item in pair[4].get("candidates", [])]
        long_prefix = [
            _candidate_signature(item)
            for item in pair[12].get("candidates", [])
            if int(item["decode_index"]) <= cutoff
        ]
        if short != long_prefix:
            prefix_mismatches += 1
    source_hashes_match = manifest.get("source_hashes") == current_hashes
    result = {
        "status_completed": manifest.get("status") == "completed",
        "manifest_counts_match": manifest.get("states") == len(states)
        and manifest.get("trials") == len(rows),
        "summary_raw_keys_match": len(raw_rows) == len(rows)
        and set(keys) == set(raw_by_key),
        "states": len(states),
        "trials": len(rows),
        "duplicate_pairs": duplicate_pairs,
        "missing_pairs": missing_pairs,
        "over_budget": over_budget,
        "prefix_mismatches": prefix_mismatches,
        "source_hashes_match": source_hashes_match,
        "source_unchanged_recorded": manifest.get("source_unchanged") is True,
    }
    result["passed"] = all(
        (
            result["status_completed"],
            result["manifest_counts_match"],
            result["summary_raw_keys_match"],
            duplicate_pairs == 0,
            missing_pairs == 0,
            over_budget == 0,
            prefix_mismatches == 0,
            source_hashes_match,
            result["source_unchanged_recorded"],
        )
    )
    return result


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit_run(run_folder: Path) -> dict:
    manifest = json.loads((run_folder / "manifest.json").read_text(encoding="utf-8"))
    with (run_folder / "trials.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    with gzip.open(run_folder / "raw_trials.jsonl.gz", "rt", encoding="utf-8") as handle:
        raw_rows = [json.loads(line) for line in handle]
    baseline = Path(manifest["baseline_repo"])
    current_hashes = {
        relative: _sha(baseline / relative) for relative in manifest["source_hashes"]
    }
    result = audit_records(manifest, rows, raw_rows, current_hashes)
    result["protocol_hash_matches"] = manifest["protocol_hash"] == _sha(
        Path(__file__).resolve().parents[1] / "protocol.json"
    )
    result["collector_hash_matches"] = manifest["script_hash"] == _sha(
        Path(__file__).with_name("action_landscape.py")
    )
    result["passed"] = (
        result["passed"]
        and result["protocol_hash_matches"]
        and result["collector_hash_matches"]
    )
    (run_folder / "audit.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    result = audit_run(args.run)
    print(json.dumps(result, ensure_ascii=False))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
