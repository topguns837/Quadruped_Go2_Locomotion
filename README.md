# Quadruped Locomotion (Isaac Lab)

Reinforcement learning project for training the Unitree Go2 quadruped in NVIDIA Isaac Lab.

## Overview

This repository provides:

- A custom Isaac Lab task: `Quadruped-Locomotion-Go2` For customised movement with extended Pitch and Lean movement of the body.  
  - The Robot is customised to include a static version of the Open Manipulator X robot mounted on its head, for compensation of the weight.
- RSL-RL training and playback scripts
- Simple validation agents (`zero_agent.py` and `random_agent.py`)
- Packaging as an Isaac Lab extension (`quadruped_go2_locomotion`)

## Compatibility

This project is tested with:

- Isaac Sim `5.1.0`
- Isaac Lab `v2.3.2`

> Isaac Lab and Isaac Sim versions must be compatible. If you use different versions, verify compatibility in the official Isaac Lab documentation.

## Installation

### 1) Install Isaac Sim + Isaac Lab

Follow the Isaac Lab source-install guide:

- https://isaac-sim.github.io/IsaacLab/v2.3.2/source/setup/installation/source_installation.html

### 2) Clone this repository

Clone this repository **outside** your Isaac Lab directory.

### 3) Install this extension package

Use the same Python environment used by Isaac Lab:

```bash
python -m pip install -e source/quadruped_go2_locomotion
```

## Robot asset and configuration

Nothing needs installing into your Isaac Lab checkout. The robot is defined locally by
`source/quadruped_go2_locomotion/.../unitree_go2witharm_cfg.py`, which resolves the USD relative to itself,
and the asset is **committed** to this repo at
`source/quadruped_go2_locomotion/.../resources/go2withArm/go2withOpenXstatic.usd`.

That asset mounts the OpenManipulator-X on the Go2's **head**, at `(0.219, 0, 0.106)` m from the base origin.
`scripts/compose_go2_with_arm.py` is reference-only and writes elsewhere; do not point the config at it. See
[docker/README.md](docker/README.md#the-go2witharm-usd-asset) for the asset's payload and remote-sublayer
dependencies, and for the error you get when the arm payload is missing.

The actuator configuration below is the part worth reading, because the calf is not the same motor as the hip
and thigh (see the comment in the snippet):

```
UNITREE_GO2WITHARM_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        # Resolved relative to the config file; see unitree_go2witharm_cfg.py for the real expression.
        usd_path=f"file://{_RESOURCES_DIR / 'go2withArm' / 'go2withOpenXstatic.usd'}",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=4, solver_velocity_iteration_count=0
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.4),
        joint_pos={
            ".*L_hip_joint": 0.1,
            ".*R_hip_joint": -0.1,
            "F[L,R]_thigh_joint": 0.8,
            "R[L,R]_thigh_joint": 1.0,
            ".*_calf_joint": -1.5,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        # Hip/thigh and calf are separate groups: the calf sits behind a 1.9169:1 knee reduction, so it is a
        # 45.43 Nm / 15.70 rad/s joint while hip and thigh are 23.7 Nm / 30.1 rad/s (Unitree's
        # go2_description.urdf, and this asset's own USD joint drives). Do not collapse these into one group
        # at 23.5 Nm -- that trains the calf at about half its real torque. Two groups are required because
        # DCMotorCfg.saturation_effort is a scalar in Isaac Lab v2.3.2.
        "hip_thigh": DCMotorCfg(
            joint_names_expr=[".*_hip_joint", ".*_thigh_joint"],
            effort_limit=23.7,
            saturation_effort=23.7,
            velocity_limit=30.1,
            stiffness=25.0,
            damping=0.5,
            friction=0.0,
        ),
        "calves": DCMotorCfg(
            joint_names_expr=[".*_calf_joint"],
            effort_limit=45.43,
            saturation_effort=45.43,
            velocity_limit=15.70,
            stiffness=25.0,
            damping=0.5,
            friction=0.0,
        ),
    },
)
"""Configuration of Unitree Go2 with addition of static Open Manipulator Mounted using DC-Motor actuator model."""
```

## Docker

As an alternative to a bare-metal Isaac Sim/Isaac Lab install, run
`./startScript.sh` for a menu-driven Docker workflow (build the image, open
an Isaac Lab shell, or an empty debugging shell; see
[docker/README.md](docker/README.md) for setup and details). The image only
bundles Isaac Sim + Isaac Lab; this repo (source, scripts, assets, `logs/`)
is bind-mounted at runtime, so editing project code never requires a
rebuild.

## Quick Start

### List available environments

```bash
python scripts/list_envs.py
```

Expected tasks include:

- `Quadruped-Locomotion-Go2`
- `Quadruped-Locomotion-Go2-Play`

### Train with RSL-RL

```bash
python scripts/rsl_rl/train.py --task=Quadruped-Locomotion-Go2
```

### Distributed training
```bash
python -m torch.distributed.run -nproc_per_node=9 ../isaaclab/isaaclab.sh -p scripts/rsl_rl/train.py --task=Quadruped-Locomotion-Go2 --headless --distributed
```

### Play a trained checkpoint

```bash
python scripts/rsl_rl/play.py --task=Quadruped-Locomotion-Go2-Play
```

### Validate environment wiring with dummy agents

Zero-action agent:

```bash
python scripts/zero_agent.py --task=Quadruped-Locomotion-Go2
```

Random-action agent:

```bash
python scripts/random_agent.py --task=Quadruped-Locomotion-Go2
```

## Development

### VS Code Python indexing (optional)

Run the workspace task `setup_python_env` to generate `.vscode/.python.env` and improve IntelliSense/Pylance module resolution for Isaac Sim/Omniverse packages.

### Code formatting

```bash
pip install pre-commit
pre-commit run --all-files
```

## Troubleshooting

### Pylance cannot resolve Isaac/Omniverse modules

Add your extension path to `.vscode/settings.json`:

```json
{
  "python.analysis.extraPaths": [
    "<path-to-repo>/source/quadruped_go2_locomotion"
  ]
}
```

### Pylance memory issues

If indexing is too heavy, reduce `python.analysis.extraPaths` entries for unused Omniverse extension groups.

## License

This project follows the license defined in the repository sources and metadata.

