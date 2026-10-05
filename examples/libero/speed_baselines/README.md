# LIBERO execution-speed baselines

This harness evaluates native, uniform-2x, SuP, and learned-precision (SAIL)
execution under one result schema. It is intentionally separate from
`examples/libero/main.py`; the earlier native smoke is setup evidence, not a
paper-protocol benchmark.

Accelerated methods use the controller and action composition supplied with the
SuP supplementary material:

- normalize the six arm dimensions into controller output space;
- sum translation and left-compose global-frame rotations;
- retain the last gripper command without multiplying it in the action merger;
- remove controller clipping, set Panda gripper speed to `0.02`, and use the
  supplied MuJoCo step implementation.

The controller patch supplies the gripper acceleration. Multiplying the gripper
again in the merger would double-compensate it and is forbidden.

Example commands (run sequentially on one GPU):

```bash
python -m examples.libero.speed_baselines.main \
  --method native --run-dir /path/to/native \
  --task-suite-name libero_10 --num-trials-per-task 1

python -m examples.libero.speed_baselines.main \
  --method uniform --run-dir /path/to/uniform2 \
  --task-suite-name libero_10 --uniform-stride 2

python -m examples.libero.speed_baselines.main \
  --method sup --run-dir /path/to/sup \
  --task-suite-name libero_10 --selector-host 127.0.0.1 --selector-port 8888

python -m examples.libero.speed_baselines.main \
  --method sail --run-dir /path/to/sail \
  --task-suite-name libero_10 --precision-threshold 0.5 \
  --sail-head-index 2 --sail-expected-tau 0.01
```

SuP requires the supplementary `predict_k` websocket service. SAIL fails closed
unless policy-server metadata declares `method=sail_precision_head`, the fast
stride matches, and inference returns one finite precision score per action.
This is the learned SAIL path; an online deviation heuristic must be labeled as
an AWE proxy and must not be reported as SAIL.

Each run holds a nonblocking lock, writes an immutable manifest, saves per-task
videos, checkpoints `results.json` atomically after every episode, and records
exact environment steps and source-action consumption. Results include an
episode-weighted overall `summary` and ordered `task_summaries` containing
success counts, rates, and successful-rollout step means. Resume is allowed only
when the full configuration matches the manifest. `--task-start` and
`--task-count` may be used for an isolated integration smoke; reportable
LIBERO-Long runs omit both restrictions and therefore cover all 10 tasks.
