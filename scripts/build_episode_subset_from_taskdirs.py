#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path

from b1k.configs.task_subset import SELECTED_TASKS


TASK_RE = re.compile(r"task-(\d+)$")
EP_RE = re.compile(r"episode_(\d+)\.json$")


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
        default="outputs/assets/task_subsets/selected12_episodes_taskindex.json",
    )
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    episodes_root = dataset_root / "meta" / "episodes"
    out_path = Path(args.output)

    if not episodes_root.exists():
        raise FileNotFoundError(f"episodes root not found: {episodes_root}")

    selected = set(int(x) for x in SELECTED_TASKS)
    episode_ids = []
    per_task_counts = {}

    for task_dir in sorted(episodes_root.iterdir()):
        if not task_dir.is_dir():
            continue

        m = TASK_RE.match(task_dir.name)
        if not m:
            continue

        task_index = int(m.group(1))
        if task_index not in selected:
            continue

        count = 0
        for ep_file in sorted(task_dir.glob("episode_*.json")):
            mm = EP_RE.match(ep_file.name)
            if not mm:
                continue
            episode_index = int(mm.group(1))
            episode_ids.append(episode_index)
            count += 1

        per_task_counts[task_index] = count

    episode_ids = sorted(set(episode_ids))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(episode_ids, f)

    print("[INFO] selected tasks:", SELECTED_TASKS)
    print("[INFO] selected episodes:", len(episode_ids))
    print("[INFO] per-task counts:", dict(sorted(per_task_counts.items())))
    print("[INFO] saved to:", out_path)


if __name__ == "__main__":
    main()
