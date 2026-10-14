# AIR_CLI — run the same training job via `databricks air`

This folder runs the **exact same** TinyLlama Ray + Lightning FSDP training workload as the
parent Databricks Asset Bundle, but submitted through the **AI Runtime CLI** (`databricks air
run`) instead of DABs. The `air` path runs as **you** (the calling user) and **streams logs
live**, which makes it ideal for fast iteration and debugging. (The DAB path deploys the job
env-to-env and runs as a service principal.)

The code is byte-identical to the bundle's: `command.sh`, `train_llama_ray_fsdp.py`,
`training_config.json`, and `training_examples.json` are verbatim copies of the DAB files.

## Files

```
AIR_CLI/
├── air_run.yaml              AI Runtime CLI run config (mirrors the DAB ai_runtime_task)
├── command.sh                Per-node entrypoint (copy of the bundle's command.sh)
├── train_llama_ray_fsdp.py   Training driver (copy of src/train_llama_ray_fsdp.py)
├── training_config.json      Training settings (copy of src/training_config.json)
└── training_examples.json    Instruction dataset (copy of src/training_examples.json)
```

## Prerequisites

- **Databricks CLI ≥ 1.20.0** with the `air` command (`databricks air --help`).
- An authenticated profile: `databricks auth login --host <workspace-url> --profile <PROFILE>`.
- AI Runtime / serverless GPU enabled in the workspace.

## Run

From inside this `AIR_CLI` folder:

```bash
# validate the config against the workspace (no submission)
databricks air run --file air_run.yaml --dry-run --profile <PROFILE>

# submit and stream logs until completion
databricks air run --file air_run.yaml --watch --profile <PROFILE>

# smaller/faster test run (single node) without editing the file
databricks air run --file air_run.yaml --watch --override compute.num_accelerators=8 --profile <PROFILE>
```

Manage runs:

```bash
databricks air list --profile <PROFILE>                 # active runs
databricks air logs <RUN_ID> --profile <PROFILE>        # stream/fetch logs (shown inline, no MLflow hunt)
databricks air get <RUN_ID> --profile <PROFILE>         # status / config / timing
databricks air cancel <RUN_ID> --profile <PROFILE>      # cancel
```

To retune training or change the data, edit `training_config.json` / `training_examples.json`
here (same keys as the bundle).

## How it maps to the DAB `ai_runtime_task`

| DAB (`databricks.yml` / `variables.yaml`) | air (`air_run.yaml`) |
|---|---|
| `ai_runtime_task.experiment` | `experiment_name` |
| `ai_runtime_task.mlflow_run` | `mlflow_run_name` |
| `accelerator_type` / `accelerator_count` | `compute.accelerator_type` / `compute.num_accelerators` |
| `base_environment: …databricks_ai_v6` | `environment.version: 6` |
| `environments[].spec.dependencies` | `environment.dependencies` |
| `environment_variables[].spec.variables` | `env_variables` |
| `deployments[].command_path` → `command.sh` | `command: bash "$CODE_SOURCE_PATH/command.sh"` |
| `artifacts.code` (packages `src/`) | `code_source.root_path` + `include_paths` |
| `task_max_retries` / `task_timeout_seconds` | `max_retries` / `timeout_minutes` |

Deliberately omitted in both: `mlflow_experiment_directory` and `mlflow_artifact_location` on a
Volume path make AI Runtime fail at task init with no logs.

## Note on environment variables

On the service-principal **DAB** job, AI Runtime applies **zero** of the `environment_variables`
(`command.sh` self-defaults via `${VAR:-default}`). The **air** run is a *user* run, so the
`env_variables` above **are** applied — but the effective values are identical to the defaults in
`command.sh`, so behavior matches either way.

## Tip

`databricks air convert-to-dabs air_run.yaml` emits a DABs bundle from this config, if you ever
want to go back the other direction.
