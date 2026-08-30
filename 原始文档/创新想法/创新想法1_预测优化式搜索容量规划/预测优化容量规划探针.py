"""E10探针：用状态—动作历史预测收益/耗时，并在时间上限内选择动作。"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path
from statistics import fmean

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression


WORKSPACE = Path(__file__).resolve().parents[1]
SOURCE = WORKSPACE / "innovation_probe" / "outputs" / "probe_06_action_results.csv"
OUTPUT = Path(__file__).resolve().parent / "outputs"
FEATURES = (
    "event_time",
    "target",
    "duration",
    "residual_count",
    "bottleneck_agv",
    "bottleneck_severity",
    "region_size",
    "budget",
)


def read_rows(path: Path = SOURCE) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    numeric = set(FEATURES) | {"dominant_improvement_rate", "wall_time"}
    for row in rows:
        for field in numeric:
            row[field] = float(row[field])
    return rows


def aggregate(rows: list[dict]) -> list[dict]:
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["state_id"], row["action_id"])].append(row)
    result = []
    for (state_id, action_id), group in sorted(groups.items()):
        first = group[0]
        result.append(
            {
                "state_id": state_id,
                "action_id": action_id,
                **{field: first[field] for field in FEATURES},
                "actual_gain": fmean(row["dominant_improvement_rate"] for row in group),
                "actual_time": fmean(row["wall_time"] for row in group),
            }
        )
    return result


def choose_action(rows: list[dict], time_limit: float) -> dict:
    feasible = [row for row in rows if row["predicted_time"] <= time_limit]
    if not feasible:
        return min(rows, key=lambda row: row["predicted_time"])
    return max(feasible, key=lambda row: (row["predicted_gain"], -row["predicted_time"]))


def _features(rows: list[dict]) -> np.ndarray:
    return np.asarray([[row[field] for field in FEATURES] for row in rows], dtype=float)


def run_probe(rows: list[dict] | None = None) -> tuple[list[dict], dict]:
    averaged = aggregate(read_rows() if rows is None else rows)
    decisions = []
    event_times = sorted({row["event_time"] for row in averaged})
    for held_out_time in event_times:
        train = [row for row in averaged if row["event_time"] != held_out_time]
        test = [row for row in averaged if row["event_time"] == held_out_time]
        gain_model = RandomForestRegressor(n_estimators=100, max_depth=4, random_state=0)
        time_model = LinearRegression()
        gain_model.fit(_features(train), [row["actual_gain"] for row in train])
        time_model.fit(_features(train), [row["actual_time"] for row in train])

        time_limit = 1.10 * fmean(row["actual_time"] for row in train if row["budget"] == 50)
        fixed_groups: dict[str, list[dict]] = defaultdict(list)
        for row in train:
            fixed_groups[row["action_id"]].append(row)
        fixed_action = max(
            (
                (action_id, fmean(item["actual_gain"] for item in group))
                for action_id, group in fixed_groups.items()
                if fmean(item["actual_time"] for item in group) <= time_limit
            ),
            key=lambda item: item[1],
        )[0]

        for state_id in sorted({row["state_id"] for row in test}):
            state_rows = [dict(row) for row in test if row["state_id"] == state_id]
            matrix = _features(state_rows)
            for row, gain, elapsed in zip(
                state_rows, gain_model.predict(matrix), time_model.predict(matrix)
            ):
                row["predicted_gain"] = float(gain)
                row["predicted_time"] = max(0.0, float(elapsed))

            selected = choose_action(state_rows, time_limit)
            actual_feasible = [row for row in state_rows if row["actual_time"] <= time_limit]
            oracle = max(actual_feasible, key=lambda row: row["actual_gain"])
            fixed = next(row for row in state_rows if row["action_id"] == fixed_action)
            decisions.append(
                {
                    "state_id": state_id,
                    "held_out_event_time": held_out_time,
                    "time_limit": time_limit,
                    "predicted_action": selected["action_id"],
                    "predicted_actual_gain": selected["actual_gain"],
                    "predicted_actual_time": selected["actual_time"],
                    "fixed_action": fixed_action,
                    "fixed_actual_gain": fixed["actual_gain"],
                    "fixed_actual_time": fixed["actual_time"],
                    "oracle_action": oracle["action_id"],
                    "oracle_actual_gain": oracle["actual_gain"],
                    "oracle_actual_time": oracle["actual_time"],
                    "prediction_beats_fixed": selected["actual_gain"] > fixed["actual_gain"],
                }
            )

    summary = {
        "cross_validation_folds": len(event_times),
        "test_state_count": len(decisions),
        "predicted_mean_gain": fmean(row["predicted_actual_gain"] for row in decisions),
        "fixed_mean_gain": fmean(row["fixed_actual_gain"] for row in decisions),
        "oracle_mean_gain": fmean(row["oracle_actual_gain"] for row in decisions),
        "prediction_wins": sum(row["prediction_beats_fixed"] for row in decisions),
        "go": fmean(row["predicted_actual_gain"] for row in decisions)
        > fmean(row["fixed_actual_gain"] for row in decisions),
    }
    return decisions, summary


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    if getattr(sys.stdout, "encoding", "").lower() != "utf-8" and hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    decisions, summary = run_probe()
    write_csv(OUTPUT / "probe_10_capacity_decisions.csv", decisions)
    write_csv(OUTPUT / "probe_10_capacity_summary.csv", [summary])
    for row in decisions:
        print(
            f"{row['state_id']}: 预测={row['predicted_action']} "
            f"固定={row['fixed_action']} Oracle={row['oracle_action']}"
        )
    print("预测策略平均改善率：", round(summary["predicted_mean_gain"], 4))
    print("固定策略平均改善率：", round(summary["fixed_mean_gain"], 4))
    print("Oracle平均改善率：", round(summary["oracle_mean_gain"], 4))
    print("容量规划探针Go：", summary["go"])


if __name__ == "__main__":
    main()
