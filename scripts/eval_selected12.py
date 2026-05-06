#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys
from pathlib import Path


SELECTED12_TASKS = [
    (0,  "turning_on_radio"),
    (1,  "picking_up_trash"),
    (5,  "setting_mousetraps"),
    (11, "putting_dishes_away_after_cleaning"),
    (12, "preparing_lunch_box"),
    (19, "outfit_a_basic_toolbox"),
    (22, "putting_shoes_on_rack"),
    (31, "clean_boxing_gloves"),
    (39, "spraying_fruit_trees"),
    (40, "make_microwave_popcorn"),
    (45, "cook_hot_dogs"),
    (46, "cook_bacon"),
]


def parse_instances(s: str) -> str:
    s = s.strip()
    if s.startswith("[") and s.endswith("]"):
        return s
    return "[" + ",".join(x.strip() for x in s.split(",") if x.strip()) + "]"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instances", default="0,1,2")
    parser.add_argument("--log-root", default="./eval_logs_selected12_3round")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default="8000")
    parser.add_argument("--only-task-id", type=int, default=None)
    parser.add_argument("--default-wrapper", action="store_true")
    args = parser.parse_args()

    repo = Path.cwd()
    eval_py = repo / "BEHAVIOR-1K" / "OmniGibson" / "omnigibson" / "learning" / "eval.py"

    if not eval_py.exists():
        raise FileNotFoundError(f"eval.py not found: {eval_py}")

    instances = parse_instances(args.instances)
    log_root = Path(args.log_root)
    log_root.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["PYTHONPATH"] = (
        f"{repo}/BEHAVIOR-1K/OmniGibson:"
        f"{repo}/BEHAVIOR-1K/bddl:"
        f"{repo}/BEHAVIOR-1K/joylo:"
        + env.get("PYTHONPATH", "")
    )

    tasks = SELECTED12_TASKS
    if args.only_task_id is not None:
        tasks = [x for x in tasks if x[0] == args.only_task_id]
        if not tasks:
            raise ValueError(f"Unknown selected12 task id: {args.only_task_id}")

    failed = []

    for task_id, task_name in tasks:
        task_log_path = log_root / f"task_{task_id:02d}_{task_name}"

        cmd = [
            sys.executable,
            str(eval_py),
            f"log_path={task_log_path}",
            "policy=websocket",
            f"task.name={task_name}",
            f"model.host={args.host}",
            f"model.port={args.port}",
            f"eval_instance_ids={instances}",
        ]

        if args.default_wrapper:
            cmd.append("env_wrapper._target_=omnigibson.learning.wrappers.DefaultWrapper")

        print("\n" + "=" * 100)
        print(f"[selected12 eval] task_id={task_id}, task_name={task_name}, instances={instances}")
        print("[cmd]", " ".join(cmd))
        print("=" * 100)

        ret = subprocess.run(cmd, env=env)
        if ret.returncode != 0:
            failed.append((task_id, task_name, ret.returncode))
            print(f"[FAILED] task_id={task_id}, task_name={task_name}, returncode={ret.returncode}")
        else:
            print(f"[DONE] task_id={task_id}, task_name={task_name}")

    print("\n" + "=" * 100)
    print("[selected12 eval finished]")
    if failed:
        print("[failed tasks]")
        for task_id, task_name, code in failed:
            print(f"  task_id={task_id}, task_name={task_name}, returncode={code}")
        sys.exit(1)
    else:
        print("All selected12 tasks finished successfully.")
    print("=" * 100)


if __name__ == "__main__":
    main()
