# src/training: how a policy is trained

**`train.sh` is the one command for training.** It reads a profile from [`profiles/`](profiles/) and runs it; everything
that defines a run (how many phases, iterations, disturbances, reward weights, PPO settings, environment count, GUI or
headless) is set in that profile. Run it from the repo root, inside the container.

## Files

| File | Purpose |
|---|---|
| `train.sh` | **The entry point.** Thin wrapper around `train_profile.py`. |
| `train_profile.py` | Reads and validates the profile, then runs one `train_batched.sh` command per phase. |
| `train_batched.sh` | Runs `train.py` in fixed-size chunks, one fresh process per chunk, in one run folder. |
| `train.py` | The worker: one RSL-RL PPO process. Not run by hand. |
| `train_mem_guard.sh`, `cleanup_old_runs.py` | Host-RAM watchdog; prune old run folders. |
| `profiles/` | Training recipes. |
| `tests/` | Profile parsing, validation and override-path tests. |

## Run a profile

Print the commands without running anything:

```bash
bash src/training/train.sh --profile obs_stacking_with_height --dry_run
```

Train:

```bash
bash src/training/train.sh --profile obs_stacking_with_height --num_envs 2048
```

## Single-phase and multi-phase profiles

Every profile starts with `num_phases`, followed by one entry per phase under `phases:`. The two must agree, or the
profile is rejected with a message naming both numbers.

| Profile | `num_phases` | What it does |
|---|---:|---|
| [`obs_stacking_with_height.yaml`](profiles/obs_stacking_with_height.yaml) | 2 | `clean` warm-up (10k iterations), then `realistic` (15k) |
| [`obs_stacking_with_height_single_phase.yaml`](profiles/obs_stacking_with_height_single_phase.yaml) | 1 | the same recipe as one `realistic` phase (25k) |
| [`iter4_realistic.yaml`](profiles/iter4_realistic.yaml) | 2 | the single-frame recipe behind the hardware-tested policy |
| [`aggressive.yaml`](profiles/aggressive.yaml) | 2 | the earlier high-disturbance recipe |
| [`example.yaml`](profiles/example.yaml) | 1 | annotated template; copy it to start your own |

To add a phase, raise `num_phases` by one and append an entry under `phases:` with its `name`, `iterations` and
`overrides`. Phase k resumes phase k-1 in the same run folder, so the curriculum and optimiser carry over.

To run only the first K phases of a profile without editing it:

```bash
bash src/training/train.sh --profile obs_stacking_with_height --num_phases 1
```

## Run settings in a profile

These top-level keys are optional and default to a normal headless run. Each has a command-line equivalent that wins
over the profile.

| Key | Meaning | Flag |
|---|---|---|
| `num_envs`, `seed` | parallel environments, random seed | `--num_envs`, `--seed` |
| `headless` | `false` opens the Isaac Sim window | `--gui` |
| `rendering_mode` | `performance`, `balanced` or `quality` | `--rendering_mode` |
| `livestream` | `1` or `2`: stream the viewport | `--livestream` |
| `max_fails` | stop after this many chunks with no new checkpoint | none |
| `dashboard`, `viewer`, `dashboard_output` | live training graphs | `--no_dashboard`, `--no_viewer`, `--dashboard_output` |

Watch a short run in a window with one environment:

```bash
bash src/training/train.sh --profile obs_stacking_with_height_single_phase --gui --num_envs 1 --rendering_mode performance
```

Resume a stopped run from a given phase:

```bash
bash src/training/train.sh --profile obs_stacking_with_height --start_phase 2
```

Any trailing `key=value` is a Hydra override applied to every phase, for example:

```bash
bash src/training/train.sh --profile obs_stacking_with_height env.rewards.track_ang_vel_z_exp.weight=3.0
```

What is trained (rewards, observations, kp/kd) is defined in `src/env/`; this folder decides how long and under what
disturbances. Hydra's per-launch dirs go to `logs/hydra/`, runs to `logs/rsl_rl/<experiment>/<run_dir>/`.
