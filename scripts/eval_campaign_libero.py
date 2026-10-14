"""Evaluate one frozen campaign unit. The external dispatcher owns GPU exclusion."""

import argparse
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8013)
    parser.add_argument("--phase-repo", type=Path)
    parser.add_argument("--preflight", action="store_true")
    opts = parser.parse_args()
    from examples.libero.speed_baselines.campaign_session import CampaignSession

    session = CampaignSession(opts.session, repo=REPO)
    if opts.preflight:
        print(
            json.dumps(
                {
                    "passed": True,
                    "physics_steps": 0,
                    "inference_calls": 0,
                    "session_sha256": session.sha256,
                    "bank": list(session.bank),
                }
            )
        )
        return
    from examples.libero.speed_baselines.main import Args
    from examples.libero.speed_baselines.main import eval_speed_baseline

    route = session.policy_spec.route
    kwargs = {}
    if route.method == "strider":
        if opts.phase_repo is None:
            parser.error("STRIDER requires --phase-repo")
        kwargs.update(
            strider_checkpoint=route.phase_checkpoint.path,
            strider_schedule=session.policy_spec.rendered_schedule.path,
            strider_phase_repo=opts.phase_repo,
            fast_stride=3,
            strider_source50_selected=True,
            strider_min_prediction_hz=session.spec.get("strider_min_prediction_hz", 0.0),
        )
    else:
        kwargs["uniform_stride"] = route.uniform_stride
    args = Args(
        method=route.method,
        run_dir=session.run_dir,
        host=opts.host,
        port=opts.port,
        task_suite_name=route.identity.suite,
        task_start=route.identity.task_id,
        task_count=1,
        num_trials_per_task=len(session.bank),
        seed=7,
        chunk_size=10,
        num_steps_wait=10,
        save_strider_telemetry=True,
        **kwargs,
    )
    try:
        eval_speed_baseline(args, campaign=session)
        from examples.libero.speed_baselines.actions import SupActionComposer
        from examples.libero.speed_baselines.campaign_audit import audit_run
        from examples.libero.speed_baselines.campaign_session import write_new

        audit = audit_run(session, composer=SupActionComposer.from_robosuite())
        write_new(session.run_dir / "terminal-audit.json", audit)
        print(json.dumps({"CAMPAIGN_AUDIT_PASSED": True, "episodes": len(audit["episodes"])}))
    finally:
        # eval_speed_baseline has synchronously exited its environment finally.
        session.close()


if __name__ == "__main__":
    main()
