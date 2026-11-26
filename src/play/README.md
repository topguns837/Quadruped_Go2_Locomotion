# src/play: run and evaluate a policy in Isaac Sim

| File | Purpose |
|---|---|
| `play.py` | Play a checkpoint (`--manual_commands` slider, `--live_plot`, `--trace_out`/`--scenario` for traces). Also exports `exported/policy.{pt,onnx}` next to the checkpoint. |
| `eval_velocity_tracking.py` | Isaac-vs-MuJoCo velocity/pitch/lean tracking comparison for one checkpoint. **Known limitation:** written for single-frame policies, so it cannot run the current stacked policy (its MuJoCo loop has no history/height support) and cannot load older single-frame checkpoints into the current stacked environment. Use `src/sim2sim/run_sim2sim_eval.sh` instead. |
| `verify_obs_stacking.py` | Meant to confirm the deploy-side observation-stack layout flags against the real env (gate before hardware). **Known issue:** it hangs while building its second environment in the same process (pass 1 finishes, pass 2 never does), so it currently cannot complete. |

```bash
isaaclab -p src/play/play.py --task=Quadruped-Locomotion-Go2-Play --num_envs 1 --checkpoint models/<run>/model_<N>.pt
```

Note: `play.py` re-exports the policy into `<checkpoint dir>/exported/`; if you point it at a committed `models/`
checkpoint, `git checkout` the exported files afterwards.
