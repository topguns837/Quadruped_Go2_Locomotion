# Plan: Depth-vision policy warm-started from the blind flat-terrain policy

## Context

We have a trained blind locomotion policy, `logs/rsl_rl/go2_with_pitch_lean_and_height_control_ppo/2026-10-14_15-26-46/model_9999.pt`. It is an MLP with input 50 → 256 → 512 → 128 → 12, plus a separate critic, and it has only seen flat terrain. The stairs env (`quadruped_go2_locomotion_stairs_env_cfg.py`) now has a D435f `depth_camera`, but nothing feeds it to the policy.

**Goal:** train a policy that combines proprioception and depth to climb stairs. Training starts from the blind weights, so on step 0 the new policy behaves exactly like the old one, and vision is learned on top of it.

Facts from research that shape the plan:
- **No image support anywhere.** Neither the installed RSL-RL (actually **3.1.2**, not 3.0.1) nor isaaclab_rl has a CNN policy or a CNN config.
  - `ActorCritic` asserts that every observation is 1D.
  - The JIT/ONNX exporters assume a single flat input to `policy.actor`.
  - What already works: the runner, PPO, `RolloutStorage` and `RslRlVecEnvWrapper` pass image tensors (N,C,H,W) through untouched, and the runner builds the policy class with `eval(class_name)`. So we need one custom module and one custom exporter, nothing more.
- **Strict loading.** `runner.load()` loads strictly and also restores the optimizer and iteration count, so warm-starting needs our own loader.
- **Observation normalization was never on.** The config's `empirical_normalization=True` is serialized as `actor_obs_normalization: {}`, which the runner treats as off. The checkpoint has no normalizer keys, so the blind weights expect **raw** observations. The new policy must keep proprio normalization off or the warm start breaks.
- **Wrong depth type for hardware.** The camera currently outputs `distance_to_camera`, which is the Euclidean distance along each ray. The real D435 reports z-depth (`distance_to_image_plane`), so this must change.
- **Camera rendered at sim rate.** The camera renders at 424×240 with `update_period=0.0`, i.e. every 5 ms physics step. That is far too expensive at 1000+ envs.

## 0. FIRST: live depth viewer in Isaac (do this before anything else)

**Why not the Camera Inspector.** It shows the viewport-style RGB render. It does not show the tensor that the `TiledCamera` actually produces, and that tensor is what the policy will consume. So we view the tensor directly.

**Plan:** a dockable Kit window built with `omni.ui`. It needs no extra dependencies and sits inside the Isaac GUI next to the Camera Inspector. Changes to `scripts/rsl_rl/play.py`:

- **New flag `--show_depth`.** It is turned on by default when `"depth_camera"` is in the scene, the same pattern as the existing frame marker at `play.py:377`.
- **Window.** After env creation, build a window `ui.Window("Depth Camera")` containing a `ui.ImageWithProvider` backed by a `ui.ByteImageProvider`.
- **Live update.** Every `--depth_view_every` steps (default 2, to keep it cheap), inside the existing loop next to the marker update at `play.py:435`:
  1. Take `camera.data.output[<depth data type>][env_idx, ..., 0]` for env `--depth_view_env` (default 0). Use whichever depth type the camera config lists, so this works both now (`distance_to_camera`) and after the switch to `distance_to_image_plane`.
  2. Replace inf/NaN with `far`, clip to [near, far] (0.28-3.0 m, from the camera's `clipping_range`), and map to a colormap on GPU: near = warm colors, far = dark, invalid pixels = magenta so holes are obvious. Convert to RGBA uint8 and call `provider.set_bytes_data(bytes, [W, H])`.
  3. The window title shows the min/max/mean depth in metres and the % of invalid pixels, as a quick numeric sanity check.
- **Later: policy view.** Once `DepthImage` (section 3) exists, a second image panel in the same window shows exactly what the policy receives: the processed and augmented `depth` observation, upscaled with nearest-neighbour, one panel per stacked frame. This is the main tool for checking the augmentations by eye.
- **Fallback if `omni.ui` misbehaves:** `--save_depth_dir` writes PNGs every N steps with `isaaclab.sensors.save_images_to_file`.

**Verify:** run `play.py --task Quadruped-Locomotion-Go2-Stairs-Play --num_envs 1`. Point the robot at stairs; the step edges should show as clear depth bands. Change `_CAMERA_MOUNT_PITCH_DEG` and confirm the view shifts.

## 1. Model architecture: `VisionActorCritic`

New file: `T/modules/vision_actor_critic.py`, where `T = source/.../tasks/manager_based/quadruped_go2_locomotion`. It subclasses `rsl_rl.modules.ActorCritic`.

```
depth (N, K=2, 60, 106) ─► DepthEncoder CNN ─► z_depth (32) ─┐
proprio "policy" group (N, 50) ──────────────────────────────┴─► concat (82) ─► actor MLP 256-512-128 ─► 12 actions
critic: [proprio 50 | height_scan 187] ─► critic MLP 256-512-128 ─► V
```

- **DepthEncoder.** Small and cheap at 1000+ envs:
  - Conv(K→16, k5, s2) → ELU
  - Conv(16→32, k3, s2) → ELU
  - Conv(32→32, k3, s2) → ELU
  - Flatten (32×8×14) → Linear 3584→128 → ELU → Linear 128→32
- **Input shape.** K=2 stacked depth frames about 0.1 s apart give short-term motion and terrain memory. Without that, the step under the front feet leaves the field of view.
- **Splitting the observations.** The class sorts the groups in `obs_groups["policy"]`: 2D tensors are concatenated as proprio, and 4D tensors go through the encoder. It overrides:
  - `get_actor_obs` and `act` / `act_inference` (the encoder runs first),
  - `update_normalization` (skip the image groups),
  - `load_state_dict`.
- **Asymmetric critic.** The critic sees privileged information: the proprio vector plus a height scan from a `RayCasterCfg` (`GridPatternCfg`, 1.6×1.0 m at 0.1 m resolution = 187 points, reusing Isaac Lab's `mdp.height_scan`).
  - The critic is thrown away at deploy time, so it can know the exact terrain. This is standard for stairs and makes value learning much faster and more stable than going through the CNN.
  - The scene gets a `height_scanner` and the env gets a `critic` observation group (proprio terms + `height_scan`). The critic does **not** see depth.
- **Pluggable encoder: `encoder_type: "small_cnn" | "efficientnet_b0"`** in the policy cfg. Train both and compare.
  - **`small_cnn`:** the 4-layer network above, trained from scratch. This is the baseline, as in ANYmal and Extreme Parkour.
  - **`efficientnet_b0` (ImageNet-pretrained, input layer replaced):**
    - **Input layer.** Build it with `timm.create_model("efficientnet_b0", pretrained=True, in_chans=K, features_only=True, out_indices=(3,))`. With `in_chans=K`, timm adapts the 3-channel RGB `conv_stem` by summing its kernels over the RGB channels and rescaling for K channels. The pretrained edge and texture filters survive this; a randomly initialized new first layer would throw them away. If timm isn't installed, fall back to torchvision's `efficientnet_b0` and do the same kernel averaging by hand.
    - **Truncate at stride 16 (stage 4, about 1M params).** The full network downsamples 32×, so a 60×106 input ends as a 2×4 map and stair-edge detail is lost. The early stages are the generic, most transferable filters and are cheaper. Upsample the input to 96×170 so the stride-16 map is 6×11. Then a 1×1 conv down to 32 channels → flatten → Linear → 32 latent. That plugs into the same actor input layout, so the zero-init warm start is unchanged.
    - **BatchNorm.** Freeze the BN layers (eval mode, fixed running statistics) so they behave as fixed affine layers. PPO computes actions in rollouts and log-probs in update minibatches; live BN statistics would make those differ, which corrupts the policy ratio.
    - **Learning rate.** Freeze the backbone for the first ~300 iterations and train only the 1×1 conv, the head and the actor. After that, unfreeze it at 0.1× the learning rate. This needs a small `PPO` subclass, because RSL-RL's adaptive schedule overwrites the learning rate of every param group; the subclass keeps the backbone group at that ratio.
    - **Weights download.** The container usually runs offline. Download the weights once with network access and save them under `T/resources/pretrained/`, then load from that file.
    - **Realistic expectation.** ImageNet features come from RGB texture, while depth is geometry, so the expected gain is mainly faster early learning of edge detectors, not a better final policy. The A/B comparison (same seed, same warm start, compare stair terrain level reached at a fixed iteration) settles it. Cost check: about 0.1 GFLOP per frame, which is fine for 1024 envs on GPU, and a few ms with TensorRT on the Go2's Jetson Orin.
  - **In-sim pretraining (optional phase 1b) works with either encoder.** Simulation gives us free labels: the ground-truth height scan.
    1. Collect about 200k (depth, height_scan) pairs by rolling out the blind policy on stairs.
    2. Train `DepthEncoder` + a small head to predict the 187-point height scan, so the encoder learns terrain geometry.
    3. Load those encoder weights before the PPO warm start.
  - The same height-scan loss can stay on during PPO as an auxiliary loss with a small weight.
- **Visualizing the network:**
  - `torchinfo.summary(policy, ...)` prints layer shapes and parameter counts.
  - Open the exported `policy.onnx` in Netron (`pip install netron; netron policy.onnx`) for an interactive graph of the encoder + actor.
  - Both are added to verification step 1.
- **Registering the class.** `T/modules/__init__.py` injects the class into the runner's namespace with `rsl_rl.runners.on_policy_runner.VisionActorCritic = VisionActorCritic`, so `class_name="VisionActorCritic"` works through the runner's `eval`. The task `__init__.py` imports this module.

## 2. Warm-start from the blind weights

New function `VisionActorCritic.load_blind_weights(path)`, called from `scripts/rsl_rl/train.py` when the new `--warm_start <ckpt>` flag is given and `--resume` isn't. It does the following:

- **Actor:** copies `actor.{2,4,6}` and `log_std` exactly.
  - `actor.0.weight` grows from 256×50 to 256×82. Columns 0-49 come from the old weights; columns 50-81 (the depth latent) are **zeros**.
  - The policy therefore outputs exactly the blind actions at iteration 0. The CNN still learns: the gradient on the zero columns is `z·δ`, which is nonzero, so they move off zero after the first update and gradient then flows back into the CNN (the ControlNet zero-init trick).
- **Critic:** same idea. Columns 0-49 are copied and columns 50-236 (height scan) start at zero. The old value function is a reasonable start, and the height-scan weights are learned.
- **CNN:** default init.
- **Optimizer and iteration:** fresh optimizer and iteration 0. `runner.load()` is never used for the warm start.
  - Ordinary `--resume` of a vision run still uses `runner.load()` unchanged, because the keys match then.
- **Proprio layout must stay identical:** same 50-dim policy group, same term order and noise, `actor_obs_normalization=False`. `load_blind_weights` asserts `actor.0.weight.shape[1] == 50` and fails loudly on a mismatch.
- **New agent config** `VisionPPORunnerCfg` in `T/agents/rsl_rl_ppo_cfg.py`:
  - `experiment_name="go2_stairs_vision"`
  - `obs_groups={"policy": ["policy", "depth"], "critic": ["critic"]}`
  - `class_name="VisionActorCritic"`, normalization off
  - `learning_rate=3e-4`, still adaptive. Lower because we're fine-tuning, and the adaptive-KL schedule protects the warm start.
  - It also clears the misleading `empirical_normalization=True`.
- **Avoiding forgetting flat ground.** The stairs terrain currently has no flat tiles. Add flat (0.15) and random_rough (0.15) back next to the four stair variants (reweighted to about 0.175 each), so the policy keeps its flat-terrain gait.

## 3. Getting depth images from the RealSense

- **Camera config** (`QuadrupedLocomotionStairsSceneCfg.depth_camera`):
  - `data_types=["distance_to_image_plane"]`, to match the real D435's z-depth.
  - Render at **106×60**, a quarter of 424×240 in each direction. The focal length and horizontal aperture stay the same, so the field of view is unchanged.
  - `update_period=0.1` (10 Hz). That cuts render cost and matches a realistic onboard depth pipeline rate.
- **New observation term** `DepthImage(ManagerTermBase)` in `T/mdp/observations.py`. It replaces the unused `depth_array`, which subsampled every 20th pixel and had no inf/NaN handling. Its params are `sensor_cfg`, `near=0.28`, `far=2.0`, `num_frames=2`, `frame_stride_steps=5`, and an augmentation config. Each call:
  1. Reads `(N,60,106,1)` and permutes it to `(N,1,60,106)`.
  2. Applies the augmentations (section 4).
  3. Sets inf, NaN and values outside [near, far] to `far`. The real D435 reports 0 for invalid pixels, and the deploy side maps those to `far` the same way.
  4. Normalizes with `(d - near)/(far - near) - 0.5`, giving the range [-0.5, 0.5].
  5. Pushes the frame into a per-env ring buffer and returns K frames spaced `frame_stride_steps` apart: `(N,K,60,106)`.
  - It resets its buffers in `reset(env_ids)`.
  - `far=2.0` rather than the 3.0 m clipping range: stairs farther than 2 m don't matter for the next step, and D435 noise grows quickly with distance.
- **New `depth` observation group** in the stairs env cfg: a single term, `concatenate_terms=True`, `enable_corruption=True` (augmentations run only when corruption is on, so `_PLAY` gets clean images).
- **Camera mount pose.** `_CAMERA_MOUNT_POS/ROT` are still placeholders, and a vision policy is tied to its camera pose. **Set the real mount pose before any training meant for hardware.** Training in sim with the placeholder is fine for getting the pipeline working.
- **Env count.** Training needs `--enable_cameras`. Start at `num_envs=1024` and drop to 512 if GPU memory or speed is a problem. Rollout storage for depth is 24×1024×2×60×106×4 B ≈ 1.25 GB, which is acceptable.

## 4. Noise and augmentations (training only, all inside `DepthImage`)

Per-env random values are drawn on reset. Values in brackets are starting points.

| Augmentation | Models | Implementation |
|---|---|---|
| Depth-dependent Gaussian noise | Stereo error grows ∝ z² | `d += N(0, σ)` with `σ = k·d²`, k ∈ [0.001, 0.005] |
| Per-env scale/offset | Depth calibration error | `d = d·(1+s) + o`, s ∈ ±2%, o ∈ ±1 cm, drawn on reset |
| Random pixel dropout | Speckle holes | p ∈ [0, 3%] of pixels → invalid (→ `far`) |
| Edge dropout | D435 holes at depth discontinuities | Pixels where the local depth gradient > threshold (0.1 m), dropped with p=0.5 |
| Random rectangular holes | Specular/IR-washed regions | 0-3 boxes per frame, up to 15×15 px, set to invalid |
| Light blur | Optics + stereo matching smoothing | 3×3 Gaussian blur with prob 0.3 |
| Latency | Capture + processing delay | Per-env delay of 0-2 policy steps (0-40 ms) applied via the ring buffer, same pattern as `DelayedJointPositionAction` in `mdp/actions.py` |
| Frame drop | Missed frames | With prob 0.05, repeat the previous frame |
| Camera extrinsics jitter | Mount tolerance | **Phase 2, optional:** a reset event that perturbs each env's camera local pose by ±1 cm / ±2° |

The proprio noise and dynamics randomization stay as they are (already tuned for the real robot), so the 50-dim stream matches what the blind policy was trained on.

## 4b. Encoder comparison: `small_cnn` vs `efficientnet_b0`

**What is compared.** We compare *whole policies*, not raw encoder outputs. Each encoder is dropped into the same `VisionActorCritic`, warm-started from the same blind MLP, and trained with PPO. We then compare how the resulting policies **behave**.
- The two latents can't be compared directly: each encoder learns its own arbitrary 32-dim feature space, and there is no ground-truth latent.
- The single exception is test 0 below, which checks the encoders against a known target, the terrain height scan.

**0. Encoder-only check, before RL (cheap, no RL noise):**
- Collect a fixed dataset of (depth, ground-truth height_scan) pairs by rolling out the blind policy on stairs. Take about 200k samples and hold out 20% for testing. This is the same data as phase 1b.
- For each encoder, train encoder + a small head to predict the 187-point height scan. Report held-out height MAE in cm, broken down by step height, plus MAE with the training augmentations applied.
- This isolates how well each encoder reads terrain geometry from depth.
- It is a fast filter and a sanity check, **not** the decision: a lower height error doesn't guarantee better stair walking. The behavioural benchmark (test 2) decides.

**Controlled runs.** Everything is identical except `encoder_type`:
- same env cfg, augmentations, PPO hyperparameters and `--warm_start` checkpoint;
- same iteration budget (e.g. 3000);
- **3 seeds each**, because RL variance between seeds is often larger than the effect we're trying to measure. Use 2 if compute is tight.
- Run names: `--run_name {encoder}_s{seed}`, so TensorBoard overlays the curves directly.

**1. Training curves (TensorBoard, already logged):**
- mean terrain level (the stair curriculum) and mean reward;
- velocity-tracking reward, to make sure vision doesn't cost tracking quality.
- Each is plotted **against iteration** (sample efficiency) and **against wall-clock time**. Wall-clock is the fair comparison if EfficientNet is slower per iteration.

**2. Fixed stair benchmark: new `scripts/rsl_rl/eval_stairs.py`, following the structure of `eval_velocity_tracking.py`:**
- **Course.** Deterministic terrain: pyramid stairs up and down at step heights {0.05, 0.10, 0.13, 0.16} m with fixed step width. 100 episodes per cell, fixed command of 0.5 m/s forward, fixed seed, deterministic actions (`act_inference`).
- **Metrics per cell:**
  - success rate: crosses the full flight without a body-contact termination;
  - average forward progress;
  - foot-collision count with step edges (`undesired_contacts`);
  - velocity-tracking error.
- **Three conditions:**
  - (a) clean depth;
  - (b) training augmentations on;
  - (c) **stress:** 2× dropout and holes, +1 extra frame of latency, ±3° camera pitch error. This tests generalization beyond the training distribution.
- **Depth ablation:** zero out the depth input in condition (a). Both encoders should get clearly worse; if one doesn't, it isn't using vision.
- **Output:** a CSV + summary table (mean ± std across seeds) per encoder.

**3. Deploy cost:**
- parameter count;
- ONNX inference latency for one frame + proprio, batch 1, measured on the training GPU now and on the Go2's Jetson with TensorRT later;
- training throughput (env-steps/s).

**Decision rule, fixed before looking at results:**
- Choose `efficientnet_b0` only if, with latency within the 5 ms budget, it shows either:
  - +5 percentage points or more stair success under the stress condition (beyond the seed std), **or**
  - the same final success while reaching the hard-stairs terrain level at least 30% sooner in wall-clock time.
- Otherwise keep `small_cnn`: simpler, faster, and nothing to download.

## 5. Export and play

- **New file** `T/modules/exporter.py` with `export_vision_policy_as_jit` and `..._onnx`. The exported module has two inputs, `proprio (1,50)` and `depth (1,K,60,106)`, and outputs actions; the encoder and actor are baked in.
- **`scripts/rsl_rl/play.py`** branches on `isinstance(policy_nn, VisionActorCritic)` to pick the exporter. The existing blind path is unchanged.
- **Out of scope here:** `deploy/deploy_real.py` (live D435 stream + identical preprocessing) and `sim2sim` (MuJoCo depth rendering, the `OBS_LAYOUT` check). These are a follow-up once a sim policy works.

## Files

- **New:** `T/modules/__init__.py`, `T/modules/vision_actor_critic.py`, `T/modules/exporter.py`
- **Modified:**
  - `T/mdp/observations.py`: `DepthImage` term, replacing `depth_array`. Also `mdp/__init__.py`.
  - `T/quadruped_go2_locomotion_stairs_env_cfg.py`: camera cfg, `height_scanner`, `depth` + `critic` groups, flat/rough tiles.
  - `T/agents/rsl_rl_ppo_cfg.py`: `VisionPPORunnerCfg`.
  - `T/__init__.py`: register `Quadruped-Locomotion-Go2-Stairs-Vision` and `-Vision-Play`, pointing at `VisionPPORunnerCfg`. The existing tasks are untouched.
  - `scripts/rsl_rl/train.py`: `--warm_start`.
  - New `scripts/rsl_rl/eval_stairs.py`: the encoder benchmark from section 4b.
  - `scripts/rsl_rl/play.py`: vision export branch.
- Update CLAUDE.md (RSL-RL version, the new task, the normalization finding) after things work.

## Verification

1. **Unit check without Kit** (`_isaac_sim/python.sh`):
   - Build `VisionActorCritic` from a fake TensorDict, load `model_9999.pt` through `load_blind_weights`, and assert that `act_inference` equals the blind `ActorCritic` output to within 1e-6 for random proprio and random depth. This proves the zero-init warm start.
   - Then run one backward pass and confirm CNN gradients are nonzero after a single optimizer step.
   - Print a `torchinfo` summary and export a dummy ONNX file to view in Netron.
2. **Env wiring:** `zero_agent.py --task Quadruped-Locomotion-Go2-Stairs-Vision --enable_cameras --num_envs 4`. Check observation shapes (policy 50, depth (4,2,60,106), critic 237), that depth has no inf/NaN, and that values stay in [-0.5, 0.5]. Save a few depth frames as PNGs, clean and augmented, and look at them.
3. **Short training smoke test:** about 50 iterations with `--warm_start` at 1024 envs. On iteration 0, mean reward and episode length should be close to the blind policy's level on easy stair tiles, not from-scratch numbers. Record iterations/s.
4. **Full training:** watch terrain level climbing over time, plus tracking rewards.
5. **Play:** the `-Vision-Play` export produces `policy.pt` and `policy.onnx`. Reload the JIT file and check it matches the in-memory policy's output.
6. **Ablation:** zero out the depth input in play. Stair success should drop noticeably, which shows the policy actually uses vision.
