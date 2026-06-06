# Quadruped Locomotion (Isaac Lab)

Reinforcement learning project for training the Unitree Go1 quadruped in NVIDIA Isaac Lab.

## Overview

This repository provides:

- A custom Isaac Lab task: `Quadruped-Locomotion-Go1` For customised movement with extended Pitch and Roll movement of the body.  
  - The Robot is customised to include a static version of the Open Manipulator X robot mounted on its head, for compensation of the weight.
- RSL-RL training and playback scripts
- Simple validation agents (`zero_agent.py` and `random_agent.py`)
- Packaging as an Isaac Lab extension (`quadruped_locomotion`)

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
python -m pip install -e source/quadruped_locomotion
```

## Quick Start

### List available environments

```bash
python scripts/list_envs.py
```

Expected tasks include:

- `Quadruped-Locomotion-Go1`
- `Quadruped-Locomotion-Go1-Play`

### Train with RSL-RL

```bash
python scripts/rsl_rl/train.py --task=Quadruped-Locomotion-Go1
```

### Play a trained checkpoint

```bash
python scripts/rsl_rl/play.py --task=Quadruped-Locomotion-Go1-Play
```

### Validate environment wiring with dummy agents

Zero-action agent:

```bash
python scripts/zero_agent.py --task=Quadruped-Locomotion-Go1
```

Random-action agent:

```bash
python scripts/random_agent.py --task=Quadruped-Locomotion-Go1
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
    "<path-to-repo>/source/quadruped_locomotion"
  ]
}
```

### Pylance memory issues

If indexing is too heavy, reduce `python.analysis.extraPaths` entries for unused Omniverse extension groups.

## License

This project follows the license defined in the repository sources and metadata.

## Citation

If this repository is helpful for your work, please consider citing:

```bibtex
@article{ranasinghe2025review,
  title={A Review of Reinforcement Learning Techniques for Quadruped Robot Control and Locomotion in Complex Terrains},
  author={Ranasinghe, Udula and Islam, Rafiqul and Anavatti, Sreenatha and Garrat, Matthew},
  journal={Available at SSRN 5183855}
}
```