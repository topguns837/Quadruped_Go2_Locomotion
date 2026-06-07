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

# Modify robots\unitree.py in IsaacLab installation


Add the following to the robots.unitree.py in IsaacLab Installation, alternately it can be defined locally, and pointing to any custom USD Model
```
UNITREE_GO2WITHARM_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"file:///home/gavin/isaacSimData/go2withArm/go2withOpenXstatic.usd",
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
        "base_legs": DCMotorCfg(
            joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
            effort_limit=23.5,
            saturation_effort=23.5,
            velocity_limit=30.0,
            stiffness=25.0,
            damping=0.5,
            friction=0.0,
        ),
    },
)
"""Configuration of Unitree Go2 with addition of static Open Manipulator Mounted using DC-Motor actuator model."""
```

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

