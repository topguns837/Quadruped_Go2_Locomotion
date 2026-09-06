# Docker setup

Runs this project's Isaac Lab task inside a container built from the official
`nvcr.io/nvidia/isaac-sim` image, with Isaac Lab (pinned to `v2.3.2`) installed
on top. The image is a **static dependency layer only**: Isaac Sim, Isaac
Lab, and their Python packages. Everything specific to this project (source
code, scripts, USD assets, `logs/`) is bind-mounted from the repo at
`docker run` time, so editing code or swapping an asset never requires a
rebuild. Only restarting the container does.

## 1. One-time host setup

### NGC account (required to pull the base image)

1. Create a free account at https://ngc.nvidia.com and sign in.
2. Go to https://catalog.ngc.nvidia.com/orgs/nvidia/containers/isaac-sim and
   click "Get Container" once. This records EULA acceptance for your account.
3. Generate an API key: your profile menu → **Setup** → **Generate API Key**.
4. Log in from this machine:
   ```bash
   docker login nvcr.io -u '$oauthtoken' -p '<YOUR_NGC_API_KEY>'
   ```

### GPU + X11 (for the Isaac Sim GUI)

- NVIDIA Container Toolkit must be installed (`docker info | grep -i nvidia`
  should list an `nvidia` runtime, it already does on this machine).
- X server access: `docker/common.sh`'s `recreate_container()` runs
  `xhost +local:docker` automatically before every container start, so you
  don't need to run it yourself. It only whitelists local Docker containers,
  not the whole network, and (like any `xhost` grant) only lasts for the
  current host X session, hence it's redone on every start rather than once.
- A native GUI window (no `--headless`, no `--livestream`) needs three things
  together in `docker-compose.yml`, and reproducibly failed with any one
  missing (`vkCreateSwapchainKHR failed` / `advanceCurrentFrame: backbuffers
  are not initialized!`, reproduced independently with `vkcube`, a minimal
  reference Vulkan app, so this isn't Isaac-Sim-specific):
  - the host's `/dev/dri` device, for Vulkan/X11 buffer sharing
  - the host's `Xauthority` file (`XAUTHORITY` must be set on the host; GDM
    sessions set it under `/run/user/<uid>/gdm/Xauthority`, not the
    traditional `~/.Xauthority`) — `xhost` alone isn't enough
  - NVIDIA Container Toolkit's `display` capability, which is separate from
    `graphics` (that one only makes the OpenGL/Vulkan libraries available,
    not X11 output access)

### The go2withArm USD asset

`source/quadruped_go2_locomotion/quadruped_go2_locomotion/tasks/manager_based/quadruped_go2_locomotion/resources/go2withArm/go2withOpenXStatic.usda`
is `.gitignore`d and is **not** in this repo (composed USD files like this are
regenerated, not versioned). Generate it once per machine by running, inside
the container:
```bash
isaaclab -p scripts/compose_go2_with_arm.py
```
This composes the standard Go2 (fetched from Nucleus) with the bundled
OpenManipulator-X asset, mounted on top of the Go2's `base` body (a sibling
in the USD hierarchy, not nested inside it — Isaac Lab's own contact-sensor
activation never recurses past a rigid body, so nesting under the rigid
`base` would make the arm's contact sensor unable to find any bodies) and
rigidly attached via a fixed joint, with collision filtering between the arm
and `base` to prevent self-collision at the mount point. It's picked up
automatically by the bind mount once generated. If it's missing, the env
will fail to spawn the robot.

## 2. Build the image

```bash
docker compose build
```

This is the slow step (~15-30 min, multi-GB download/build) and only needs
to be re-run when you change the Isaac Sim/Isaac Lab version pins in
`docker-compose.yml`, not when you change project code.

`docker/Dockerfile` is a two-stage build:

- **`base`**: everything expensive and one-time (apt packages, Isaac Lab
  source, every pip install including the slow `isaaclab.sh --install`).
  Only changes when the version pins change.
- **`app`** (`FROM base`): cheap, fast-to-rebuild bits on top (bashrc/env
  aliases, `entrypoint.sh`). `docker compose build` builds this stage; it
  reuses the cached `base` layers unless they've changed, so rebuilding
  after only touching `docker/entrypoint.sh` takes seconds, not minutes.

To build just the base layer standalone (e.g. to warm a cache):

```bash
docker build -f docker/Dockerfile --target base -t quadruped-go2-base ./docker
```

`base` also carries a fix worth knowing about: a bare `isaaclab.sh --install`
silently fails to install the core `isaaclab` package, because one of its
dependencies, `flatdict==4.0.1`, ships a legacy `setup.py` that needs
`pkg_resources` at build time, and a freshly-fetched `setuptools` no longer
bundles it. The Dockerfile pins `setuptools<81` via `PIP_CONSTRAINT` before
that step to keep `pkg_resources` available. If a future Isaac Lab version
bump reintroduces a similar legacy-sdist dependency, this is the mechanism to
reach for again.

## 3. Run

From the repo root:

```bash
./startScript.sh
```

This shows a menu:

1. **Open Isaac Lab shell**, runs `docker/isaaclab-shell.sh`: opens a tmux
   session with three windows (`list-envs`, `train`, `play`), each with its
   command already typed into the pane but **not executed**. Press Enter on
   whichever one you want, or edit it first:
   ```
   list-envs:  isaaclab -p scripts/list_envs.py
   train:      isaaclab -p scripts/rsl_rl/train.py --task=Quadruped-Locomotion-Go2 --rendering_mode performance --num_envs 1 --livestream 2
   play:       isaaclab -p scripts/rsl_rl/play.py --task=Quadruped-Locomotion-Go2-Play --num_envs 1 --rendering_mode performance --livestream 2
   ```
   Switch windows with `Ctrl-b` then a number (`0`/`1`/`2`), or `Ctrl-b n`/`p`.
2. **Build docker image**, runs `docker/build.sh` (`docker compose build`).
3. **Open empty container**, runs `docker/shell.sh`: a bare tmux prompt at
   `/workspace/quadruped_go2_locomotion` (this repo, bind-mounted, with the
   `quadruped_go2_locomotion` extension already `pip install -e`'d), for
   free experimentation/debugging.
4. **Exit**.

**Important, container lifecycle:** options 1 and 3 always kill and recreate
the project's single container first (`docker compose down` then `up -d`),
even if you're reopening the same activity that's already running. Anything
that was running inside the previous container (e.g. an in-progress training
job) is lost. Detach from tmux with `Ctrl-b d` to leave a session running in
the background instead of exiting it, but note that choosing a menu option
again will still kill it, since a new option always recreates the container.

Checkpoints under `logs/` land directly on your host since the whole repo is
bind-mounted, so they survive container recreation regardless.

`isaaclab` is a bash alias baked into the image for `$ISAACLAB_PATH/isaaclab.sh`;
`python`/`pip` aliases point at the Isaac Sim Python env the same way. These
aliases only exist in interactive shells. For a scripted one-off command
against a container already started via `./startScript.sh`, use the full path:

```bash
docker compose exec quadruped-go2 /workspace/isaaclab/isaaclab.sh -p scripts/zero_agent.py --task=Quadruped-Locomotion-Go2
```

## Notes

- `network_mode: host` is used (matches Isaac Lab's own docker setup). It
  simplifies X11 and avoids Vulkan/EGL/streaming port-mapping issues.
- GPU VRAM: this was set up against a laptop RTX 3060 with only 6GB VRAM,
  shared with the desktop itself (single-GPU laptop, no separate display
  GPU). The GUI's default RTX rendering can exhaust that, and because the
  desktop shares the same GPU, a renderer hang can freeze the whole host, not
  just the container. The pretyped `train`/`play` commands in
  `docker/isaaclab-shell.sh` use two mitigations together:
  - `--rendering_mode performance`: a lighter real-time preset, no path
    tracing.
  - `--livestream 2`: instead of opening a native X11 window, this runs
    Isaac Sim's lighter headless-render experience and streams the viewport
    over WebRTC. It skips loading the full interactive Editor's window-chrome
    extensions (menus, drag-drop, hotkeys), which is real overhead, not just
    a smaller GPU footprint. Watch the tmux pane for the URL it prints (a
    `localhost` address) and open that in a browser to see the viewport.
    Remove `--livestream 2` from the pretyped command if you want the native
    window back (e.g. on a machine with more VRAM headroom); it's just typed
    text in the tmux pane, not a hardcoded default.
  - If it still freezes: add `--num_envs` reductions too (see
    `quadruped_go2_locomotion_env_cfg.py`), since this is a hard VRAM
    ceiling, not a bug.
  - **Recovering from a freeze without a hard power-cycle**: try switching
    to a text VT first (`Ctrl+Alt+F3` or `F4`), logging in, and running
    `docker kill quadruped_go2_locomotion` to force-stop whatever is holding
    the GPU. VT switching is a kernel-level operation and often still works
    even when Xorg/the GPU renderer is hung, so this can let the desktop
    recover without losing unrelated work. If the VT switch itself doesn't
    respond, the GPU driver itself is wedged and a hard reset is the only
    option.
- Isaac Sim shader/kit caches persist in named Docker volumes
  (`isaac-cache-*`). These survive the kill-and-recreate cycle (only
  `docker compose down -v` wipes them), so the first GUI launch is slow
  (shader compilation) but later ones stay fast.
