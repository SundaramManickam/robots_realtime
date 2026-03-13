# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Robots Realtime is a modular Python framework for realtime robot control, teleoperation, and policy integration. It supports bimanual I2RT YAM arms, Franka Panda, and MuJoCo simulation environments. The system runs agents in a control loop at configurable Hz rates (typically 30Hz), streaming joint commands to physical or simulated robots.

## Build & Run Commands

```bash
# Install (requires Python 3.11, uses uv package manager)
uv venv --python 3.11
uv pip install -e .

# Launch with a config (primary entrypoint)
uv run robots_realtime/envs/launch.py --config_path configs/yam/yam_viser_bimanual.yaml

# Install optional extras
uv sync --extra sensors --extra franka_panda    # Franka + ZED cameras
uv sync --extra mjlab_sim                        # MuJoCo sim via mjlab
uv sync --extra quest_vr                         # Quest VR teleoperation

# Linting (ruff, configured in pyproject.toml)
ruff check                  # lint
ruff check --fix            # lint + autofix
ruff format                 # format
```

## Architecture

### Config-Driven Instantiation
The system uses a Hydra-like `_target_` pattern for config-driven object creation. YAML configs specify `_target_: fully.qualified.ClassName` with constructor args as sibling keys. The `instantiate()` function in `robots_realtime/envs/configs/instantiate.py` recursively resolves these. Configs are loaded via `DictLoader` (OmegaConf-based) from `robots_realtime/envs/configs/loader.py`.

### Launch Flow (`robots_realtime/envs/launch.py`)
The single entrypoint parses a YAML config into a `LaunchConfig` and runs one of two paths:
- **Real hardware mode**: Sets up CAN interfaces, spawns robots and agents as separate processes communicating via Portal RPC, creates a `RobotEnv` (dm_env interface), and runs the control loop.
- **Sim mode** (`sim_mode: true`): Instantiates everything in-process (no Portal RPC), runs a simplified control loop with MuJoCo viewer on the main thread.

### Portal RPC Layer (`robots_realtime/utils/portal_utils.py`)
Robots and agents run in separate processes. `RemoteServer` wraps any object, binding methods marked with `@remote()` or listed in a custom methods dict. `Client` provides a transparent proxy that calls these methods over the wire. The `return_futures` context manager enables async batching of RPC calls for parallel sensor/robot reads.

### Key Abstractions
- **`Robot`** (from `i2rt.robots.robot`): Interface with `get_joint_pos()`, `command_joint_pos()`, `get_observations()`, `num_dofs()`. Robot configs live in `robot_configs/` as YAML files with `_target_` pointing to concrete implementations.
- **`Agent`** (Protocol in `robots_realtime/agents/agent.py`): Must implement `act(obs) -> action_dict` and `action_spec()`. Actions are dicts keyed by robot name containing `{"pos": np.ndarray}`.
- **`RobotEnv`** (`robots_realtime/envs/robot_env.py`): dm_env wrapper that orchestrates step/reset/obs across multiple robots and cameras.
- **`ConcatenatedRobot`** (`robots_realtime/robots/robot.py`): Merges multiple robot chains into one logical robot with optional joint remapping.

### Config Structure
- `configs/` — Launch configs (top-level YAML with LaunchConfig target, agent, sensors, robots)
- `robot_configs/` — Per-robot hardware configs (referenced by path from launch configs, support layered overrides via list of YAML files)
- Robot entries in launch configs can be a list of YAML paths that get merged in order (later files override earlier ones)

### Submodules (`dependencies/`)
Four git submodules installed as editable packages:
- **i2rt** — CAN motor drivers, robot kinematics, motor chain robot abstraction
- **pyroki** — JAX-based inverse kinematics solver
- **jaxls** — JAX nonlinear least squares (pyroki dependency)
- **mjlab** — MuJoCo simulation environment builder

### Agent Types
- `agents/teleoperation/` — Viser-based IK teleoperation, GELLO leader arm, bilateral control, Quest VR
- `agents/policy_learning/` — Diffusion Policy, PI0 policy agents
- `agents/client/` — Client-side agents connecting to external IK servers (FastAPI in `serving/`)

## Code Style
- Ruff linter with 119-char line length, configured in `pyproject.toml`
- Type annotations expected (ANN rules enabled)
- `dependencies/` directory is excluded from linting and type checking
- Python 3.11 only (hard pinned `>=3.11,<3.12`)
