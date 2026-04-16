#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import pandas as pd

from b1k.configs.task_subset import SELECTED_TASKS


def first_task(x):
    if isinstance(x, list):
        return x[0] if x else None
    return x


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root",
        type=str,
        default="/home/data/datasets/behavior_224_rgb",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/assets/task_subsets/selected12_episodes.json",
    )
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    episodes_path = dataset_root / "meta" / "episodes.jsonl"
    tasks_path = dataset_root / "meta" / "tasks.jsonl"
    out_path = Path(args.output)

    ep = pd.read_json(episodes_path, lines=True)
    tk = pd.read_json(tasks_path, lines=True)

    if "episode_index" not in ep.columns or "tasks" not in ep.columns:
        raise RuntimeError(f"episodes.jsonl columns unexpected: {list(ep.columns)}")
    if "task_index" not in tk.columns or "task" not in tk.columns:
        raise RuntimeError(f"tasks.jsonl columns unexpected: {list(tk.columns)}")

    ep = ep.copy()
    ep["task_text"] = ep["tasks"].map(first_task)

    tk = tk.copy()
    tk["task_text"] = tk["task"]

    merged = ep.merge(
        tk[["task_index", "task_name", "task_text"]],
        on="task_text",
        how="left",
    )

    if merged["task_index"].isna().any():
        missing = merged[merged["task_index"].isna()]["task_text"].dropna().unique().tolist()
        print("[WARN] unmatched task texts (first 20):", missing[:20])

    merged = merged.dropna(subset=["task_index"])
    merged["task_index"] = merged["task_index"].astype(int)
    merged["episode_index"] = merged["episode_index"].astype(int)

    filtered = merged[merged["task_index"].isin(SELECTED_TASKS)]
    episode_ids = sorted(filtered["episode_index"].unique().tolist())

    per_task_counts = (
        filtered.groupby("task_index")["episode_index"]
        .nunique()
        .sort_index()
        .to_dict()
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(episode_ids, f)

    print("[INFO] selected tasks:", SELECTED_TASKS)
    print("[INFO] selected episodes:", len(episode_ids))
    print("[INFO] per-task unique episode counts:", per_task_counts)
    print("[INFO] saved to:", out_path)


if __name__ == "__main__":
    main()
