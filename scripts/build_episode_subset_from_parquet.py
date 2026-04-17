#!/usr/bin/env python3
import json
import re
from pathlib import Path

import pandas as pd

from b1k.configs.task_subset import SELECTED_TASKS

DATA_ROOT = Path("/home/data/datasets/behavior_224_rgb/data")
OUT_PATH = Path("outputs/assets/task_subsets/selected12_episodes_parquet_verified.json")

EP_RE = re.compile(r"episode_(\d+)\.parquet$")

selected = set(int(x) for x in SELECTED_TASKS)
valid_episode_ids = []
mismatches = []

for task_id in SELECTED_TASKS:
    task_dir = DATA_ROOT / f"task-{task_id:04d}"
    if not task_dir.exists():
        print(f"[WARN] missing dir: {task_dir}")
        continue

    for pq in sorted(task_dir.glob("episode_*.parquet")):
        m = EP_RE.match(pq.name)
        if not m:
            continue
        episode_index = int(m.group(1))

        try:
            df = pd.read_parquet(pq, columns=["task_index"])
        except Exception as e:
            mismatches.append({
                "episode_index": episode_index,
                "task_dir": task_id,
                "reason": f"read_error: {e}",
            })
            continue

        uniq = sorted(set(int(x) for x in df["task_index"].dropna().unique().tolist()))

        # 실제 parquet 안 task_index가 전부 선택된 task이고, 현재 task_dir와도 일치할 때만 통과
        if uniq == [task_id]:
            valid_episode_ids.append(episode_index)
        else:
            mismatches.append({
                "episode_index": episode_index,
                "task_dir": task_id,
                "task_index_values": uniq,
            })

valid_episode_ids = sorted(set(valid_episode_ids))

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(OUT_PATH, "w", encoding="utf-8") as f:
    json.dump(valid_episode_ids, f)

print("[INFO] selected tasks:", SELECTED_TASKS)
print("[INFO] verified episodes:", len(valid_episode_ids))
print("[INFO] saved to:", OUT_PATH)
print("[INFO] mismatches:", len(mismatches))

for item in mismatches[:30]:
    print("[MISMATCH]", item)
