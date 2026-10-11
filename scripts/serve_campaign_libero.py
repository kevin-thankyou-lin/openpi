"""Serve the pinned campaign pi0.5 policy, or preflight with no GPU imports."""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def main() -> None:
    from examples.libero.speed_baselines.campaign_policy import load_policy_spec
    from examples.libero.speed_baselines.campaign_policy import preflight
    from examples.libero.speed_baselines.campaign_policy import serve

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, type=pathlib.Path)
    parser.add_argument("--port", type=int, default=8013)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    spec = load_policy_spec(args.spec)
    if args.preflight:
        print(json.dumps(preflight(spec, repo=REPO), sort_keys=True))
    else:
        serve(spec, repo=REPO, port=args.port)


if __name__ == "__main__":
    main()
