# src/play: run and evaluate a policy in Isaac Sim

| File | Purpose |
|---|---|
| `play.py` | Play a checkpoint (`--manual_commands` slider, `--live_plot`, `--trace_out`/`--scenario` for traces). Also exports `exported/policy.{pt,onnx}` next to the checkpoint. |
| `eval_velocity_tracking.py` | Isaac-vs-MuJoCo velocity/pitch/lean tracking comparison for one checkpoint (works with single-frame and stacked policies; pass the matching `--mujoco_config` deploy profile). `src/sim2sim/run_sim2sim_eval.sh` compares seeded random command schedules instead. |
| `verify_obs_stacking.py` | Confirms the deploy-side observation-stack layout flags against the real env (gate before hardware): runs the env once with history 1 and once stacked, each in its own process, and checks `deploy_real.ObsHistory` reproduces the env's vector. Prints PASS/FAIL. |

```bash
isaaclab -p src/play/play.py --task=Quadruped-Locomotion-Go2-Play --num_envs 1 --checkpoint models/<run>/model_<N>.pt
```

Note: `play.py` re-exports the policy into `<checkpoint dir>/exported/`; if you point it at a committed `models/`
checkpoint, `git checkout` the exported files afterwards.
