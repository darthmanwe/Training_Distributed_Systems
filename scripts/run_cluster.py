"""Multi-node Ray cluster launcher (requires Linux/Docker for full functionality)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch distributed training on a Ray cluster")
    parser.add_argument("--head-address", type=str, required=True, help="Ray head node address")
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()

    import ray

    ray.init(address=args.head_address)

    from scripts.run_local import run_training
    from src.infra.config import load_config

    config = load_config(args.config, overrides=args.overrides if args.overrides else None)
    run_training(config, resume_path=args.resume)


if __name__ == "__main__":
    main()
