"""Utility to manually kill rollout workers for demo/testing purposes."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ray


def main() -> None:
    parser = argparse.ArgumentParser(description="Kill rollout workers")
    parser.add_argument("--num", type=int, default=1, help="Number of workers to kill")
    parser.add_argument("--random", action="store_true", help="Kill random workers")
    args = parser.parse_args()

    if not ray.is_initialized():
        ray.init(address="auto")

    actors = ray.util.list_named_actors()
    worker_actors = [a for a in actors if a.startswith("worker")]

    if not worker_actors:
        print("No worker actors found.")
        return

    to_kill = args.num
    if args.random:
        targets = random.sample(worker_actors, min(to_kill, len(worker_actors)))
    else:
        targets = worker_actors[:to_kill]

    for name in targets:
        try:
            handle = ray.get_actor(name)
            ray.kill(handle)
            print(f"Killed worker: {name}")
        except Exception as e:
            print(f"Failed to kill {name}: {e}")


if __name__ == "__main__":
    main()
