# Repo health check

One command that answers "is this commit usable?": it starts a **separate, throwaway container**, runs the basics of
every stage (profiles, training, export, play, MuJoCo sim2sim, Isaac-vs-MuJoCo eval, deploy against a synthetic
robot), measures code coverage, and writes a report. Your main container is never touched.

## Run it

From the repo root on the host (Docker and an NVIDIA GPU needed; the first run also compiles Isaac Sim shaders, so it is
slower):

```bash
./healthcheck.sh
```

Check one specific commit (it is checked out into a temporary git worktree, so your working tree is not disturbed):

```bash
./healthcheck.sh --ref 3f2a9c1
```

Run only some groups or cases (comma separated group names, case ids or id prefixes; dependencies are added
automatically). Coverage is not enforced for partial runs:

```bash
./healthcheck.sh --only sim2sim,deploy
```

List the cases without running anything:

```bash
./healthcheck.sh --list
```

Other options: `--keep` keeps the container and training outputs for debugging, `--no_coverage` skips coverage,
`--gpu_wait_min N` sets how long to wait for a busy GPU (default 20) before skipping the Isaac cases,
`--fail_under N` changes the coverage threshold (default 90).

The report is `logs/healthcheck/<timestamp>_<commit>/report.md` (plus `junit.xml`, one log per case, and
`htmlcov/index.html`). Exit code 0 means every case passed and coverage is at or above the threshold.

## What it runs

| Group | What it proves | Needs GPU |
|---|---|---|
| `hygiene` | pre-commit passes (run on the host), the extension imports in a fresh container, README/doc links resolve, referenced files exist, committed policies load and match their deploy profile | no |
| `unit` | every unit test under `src/*/tests` | no |
| `profiles` | every shipped training profile resolves; `--num_phases`, `--num_envs`, `--gui` and Hydra overrides reach the command; each kind of malformed profile is rejected with its own message | no |
| `env` | both tasks are registered and step with zero and random actions | yes |
| `training` | a 1-phase and a 3-phase profile train, chunk seams resume, the curriculum is restored at every seam, per-phase overrides are applied, `--num_phases` / `--start_phase` work, a finished run is not retrained, a bad Hydra key fails loudly | yes |
| `play` | a fresh checkpoint plays, is exported (JIT + ONNX, 288 inputs), and records a trace; the stacked-observation layout matches `deploy_real.ObsHistory`; the velocity-tracking evaluation runs | yes |
| `sim2sim` | every deploy profile on every scenario in MuJoCo; noise, delay, low-pass, speed-cap, gain overrides; history-length mismatches fail loudly; the Isaac-exported policy runs in MuJoCo; Isaac and MuJoCo traces compare; the eval script writes a finite report | partly |
| `deploy` | the 2 deploy x 5 safety profile matrix in dry run; `deploy_real.py` against a synthetic robot on loopback (valid CRCs); e-stop, joint-speed fault and link loss each stop and damp; pre-flight check; wireless relay over loopback; deploy traces turn into scenarios and analysis reports | no |

## How it is isolated

- The container is `quadruped_go2_locomotion_testing`, defined under the `testing` profile of
  [`docker-compose.yml`](../../docker-compose.yml) so `./start.sh` never starts it. It uses the same image, its own
  network namespace (so DDS traffic on `lo` and process kills cannot reach the host, your main container or a robot),
  no display, and its own Isaac caches.
- It shares the GPU. If another job (for example real training) leaves less than 8 GB of VRAM free, the runner waits,
  then skips the GPU cases instead of slowing that job down.
- Training outputs are written under `logs/rsl_rl/healthcheck_*` and deleted afterwards unless `--keep` is given.

## Test profiles

Everything the health check feeds the real code lives in [`profiles/`](profiles/), separate from the profiles you train
and deploy with. Each file has one deliberate edge case, named in its header comment.

| Folder | Profile | Edge case |
|---|---|---|
| [`training/`](profiles/training/) | `single_phase.yaml` | the simple case: 4 iterations, one chunk seam |
| | `three_phase.yaml` | 3 phases (2 + 3 + 2 iterations), a different push velocity per phase |
| | `unbatched.yaml` | `batched: false`, the whole phase in one process |
| | `one_iteration.yaml` | the smallest possible run |
| | `uneven_batches.yaml` | 5 iterations in chunks of 2, so the last chunk is partial |
| | `overrides_mix.yaml` | env and agent overrides at top level and per phase, per-phase seeds, a zero-weight reward |
| | `envs_change.yaml` | phase 2 uses a different `num_envs` than phase 1 |
| | `fast_curriculum.yaml` | the difficulty curriculum is forced to move every ~2 iterations, then saved and restored across a seam |
| | `run_settings.yaml` | dry-run only: `headless: false`, `rendering_mode`, `livestream`, `max_fails` |
| | `bad_*.yaml` (9 files) | each malformed in one way: phase count, missing/zero/duplicate fields, baked-in constant, bad head, unknown key, non-existent Hydra key |
| [`deploy/`](profiles/deploy/) | `stacked_no_filter.yaml`, `stacked_strong_filter.yaml`, `stacked_nyquist_filter.yaml` | joint-velocity low-pass off, at 1 Hz, and at the Nyquist rate |
| | `stacked_low_gains.yaml` | PD gains below what the policy trained with |
| | `stacked_slow_link_guard.yaml` | a 2 s link-loss threshold |
| | `single_with_filter.yaml` | the old single-frame policy with the filter switched on |
| | `bad_*.yaml` (5 files) | history-length mismatch, missing policy file, 11-joint order, a list instead of a mapping, an empty file |
| | `safety_test.yaml` | safety profiles `tight`, `slow_start`, `lenient`, and `typo_key` (a misspelled key that must be rejected) |
| [`scenarios/`](profiles/scenarios/) | `short.yaml` | the simple case: stand, walk, turn |
| | `zero_command.yaml`, `limits.yaml`, `rapid_reversal.yaml`, `five_dim_command.yaml` | stand still; every command at its trained limit; sign flips every 0.5 s; commands without a height value |
| | `bad_*.yaml` (3 files) | no segments, a 3-value command, a negative duration |

The profile-by-profile combinations (deploy profile x safety profile, scenario x deploy profile) are generated in
[`cases.yaml`](cases.yaml).

## Coverage

Every Python process the cases start (including `train.py` under Isaac Sim) records coverage through a `.pth` hook
installed in the throwaway container only; the runner combines the data at the end. The run finishes by printing a
summary (total, the least-covered files, where the reports are) and writes, into the run folder:

| File | Content |
|---|---|
| `htmlcov/index.html` | browsable per-line report: open it in a browser |
| `coverage.txt` | per-file table with the line numbers that were never run |
| `coverage.xml` | Cobertura XML for CI tools |
| `coverage.json` | raw data |
| `report.md` | the pass/fail table plus the per-file coverage table |

The configuration omits the vendored MuJoCo Menagerie tree, tests, and the tools that need physical hardware (listed under
`coverage_omit` in `cases.yaml`: mode/lowstate probes, single-joint test, joint-mapping verification, still-observation
collection, UDP probe). The run fails if total coverage is below `--fail_under` (default 90).

## Adding a case

Add an entry to [`cases.yaml`](cases.yaml). The commonly used keys are `id`, `group`, `cmd`, `timeout`, `gpu`, `needs`,
`expect` (exit code and log patterns), `artifacts` and `checks` (functions in [`asserts.py`](asserts.py)); the header of
the file lists all of them. Use `matrix:` to generate one case per combination. A case that exits 0 fails if its log
contains a Python traceback. Mark a documented problem with `known:` and it is reported without failing the run.

Tips: end background processes by process group (see [`fake_robot_case.sh`](fake_robot_case.sh)); never `pkill -f` a
pattern that also appears in your own command line.

## Files

| File | Role |
|---|---|
| [`../../healthcheck.sh`](../../healthcheck.sh) | host launcher: starts and removes the testing container, runs pre-commit on the host |
| [`run.py`](run.py) | runner: executes cases, enforces expectations, collects coverage, writes reports |
| [`cases.yaml`](cases.yaml) | the case list |
| [`asserts.py`](asserts.py) | Python checks used by cases |
| [`check_repo.py`](check_repo.py) | static consistency checks (links, paths, policies, profiles) |
| [`fake_robot_case.sh`](fake_robot_case.sh) | runs one deploy session against the synthetic robot and cleans up |
| [`tests/`](tests/) | unit tests for the runner itself |
