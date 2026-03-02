# Submission (HW2)

This folder contains my Homework 2 code submission for **Robot Learning (2026)**.

The main content is the `hw2/` package (training + planning + evaluation for SimpleWorldModel and DreamerV3 on LIBERO).

## Contents

- `hw2/` — HW2 source code structure
	- `dreamer_model_trainer.py` — main entrypoint used for most experiments
	- `planning.py` — CEM planner + policy-guided planning
	- `simple_world_model.py` — pose-based simple world model
	- `dreamerV3.py`, `networks.py` — DreamerV3 world model + neural nets
	- `sim_eval.py` — evaluation / rollout utilities
	- `conf/` — Hydra configs (model/planner/dataset)
	- `scripts/` — helper shell scripts for running experiments

## Setup

This submission assumes the course environment (LIBERO + MuJoCo + dependencies) is available.

## How to run

Most commands should be run from `submit/hw2/`.

### 1) SimpleWorldModel + CEM planning (Part 1)

Run training + evaluation using the simple pose world model and CEM planning:

```bash
./scripts/part_1_default.sh
```

### 2) Policy training + policy-guided CEM (Part 2)

Train a policy (offline data), then use it to initialize CEM:

```bash
cd submit/hw2

# Train the policy
./scripts/part_2_offline_data_training.sh

# Policy-guided CEM
./scripts/part_2_policy_guided_cem.sh  
```

### 3) DreamerV3 planning (Parts 3)

Dreamer experiments are configured through the same entrypoint and Hydra configs.
Check `hw2/conf/` for the exact settings used.

Common pattern:

```bash
./scripts/part_3_default.sh
```

### 4) Image-Based Policy training + policy-guided CEM (Part 4)

Train a policy (offline data), then use it to initialize CEM:

```bash
cd submit/hw2

# Train the policy
./scripts/part_4_offline_data_training.sh

# Policy-guided CEM
./scripts/part_4_policy_guided_cem.sh  
```

## Outputs

Runs typically write artifacts under `submit/hw2/outputs/` (Hydra default) including:

- logs (training curves / metrics)
- checkpoints (world model / policy)
- optional videos (if enabled)

## Troubleshooting

- **Hydra config errors**: make sure you run from `submit/hw2/` so relative config paths resolve.
- **LIBERO not found / MuJoCo errors**: ensure the course environment (LIBERO + MuJoCo) is installed and environment variables are set.
- **GPU / CUDA**: DreamerV3 is much faster with a GPU; SimpleWorldModel should run fine on CPU.

## Notes for graders

- The main entrypoint is `submit/hw2/dreamer_model_trainer.py`.
- Planner implementations are in `submit/hw2/planning.py`.
- All Hydra configs are under `submit/hw2/conf/`.
