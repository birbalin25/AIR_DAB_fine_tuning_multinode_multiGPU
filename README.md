# Fine-Tuning TinyLlama on Databricks AI Runtime

A Databricks Asset Bundle (DAB) that fine-tunes **TinyLlama** across multiple GPU nodes on
Databricks **AI Runtime** (serverless GPU, Public Preview). It uses **Ray Train** to launch one
worker per GPU and **PyTorch Lightning FSDP** (`RayFSDPStrategy`) for sharded distributed
training. The whole job — compute, environment, code, and schedule — is defined declaratively and
deployed with the Databricks CLI.

> For architecture depth and the hard-won invariants, see [`CLAUDE.md`](./CLAUDE.md). Read it
> before changing `command.sh`, the AI Runtime task config, or the resume logic.

## How it works

1. AI Runtime provisions the GPU nodes (for example, 24 GPUs = 3 × `GPU_8xH100`) and builds the
   Python environment.
2. It runs `command.sh` on every node. Node 0 starts the Ray head and the training driver; the
   other nodes join as Ray workers.
3. The driver waits for all GPUs to join, then starts Ray Train, which runs the training function
   once per GPU with Lightning FSDP.
4. Checkpoints are written to a Unity Catalog Volume, and the cluster tears down cleanly when the
   driver finishes.

## Repository layout

```
air_ray_fsdp_llama_dabs/
├── databricks.yml            Bundle definition: job, AI Runtime task, compute, targets, permissions
├── variables.yaml            Deployment & infrastructure variables; per-environment overrides
├── command.sh                Per-node entrypoint: starts Ray, launches the driver
├── src/                      Packaged into dist/code.tgz and shipped to every node
│   ├── train_llama_ray_fsdp.py   Training driver (Ray Train + Lightning FSDP)
│   ├── training_config.json      All training settings + resume_checkpoint (edit to retune)
│   └── training_examples.json    Instruction-tuning dataset
└── CLAUDE.md                 Engineering notes & invariants
```

## Prerequisites

- **Databricks CLI ≥ 1.20.0** (`databricks --version`). Earlier versions silently drop the
  `ai_runtime_task` fields.
- A workspace with AI Runtime / serverless GPU enabled.
- A Unity Catalog Volume for checkpoints.
- A service principal to run the job (recommended for production).
- An authenticated CLI profile: `databricks auth login --host <workspace-url> --profile <PROFILE>`.

## Configure for your workspace

| Where | Setting | Example |
|---|---|---|
| `databricks.yml` (targets) | `workspace.host` — workspace URL (literal; interpolation not allowed) | `https://<your-workspace>.cloud.databricks.com` |
| `variables.yaml` (per target) | `workspace_root_path` — folder the bundle deploys into | `/Workspace/prod` |
| `variables.yaml` (per target) | `command_path` — must equal `<root_path>/files/command.sh` | `/Workspace/prod/files/command.sh` |
| `variables.yaml` (per target) | `service_principal_application_id` — the SP the job runs as | `<service-principal-application-id>` |
| `src/training_config.json` | `ray_storage_path` — UC Volume for checkpoints | `/Volumes/<catalog>/<schema>/<volume>/ray_results` |

Then grant:

- The service principal **`WRITE_VOLUME`** on the checkpoint Volume (so it can write checkpoints).
- The deploying identity the **`servicePrincipal.user`** role on the SP (so `run_as` binds at
  deploy time).
- Keep the top-level **`permissions:`** block — it grants the SP (and the deployer) `CAN_MANAGE`
  so the job, running as the SP, can read the deployed code. This is required for any non-shared
  `root_path` (for example, under `/Workspace/...`); without it the task fails with
  `User lacks read access to file: .../command.sh`.

## Configuration: two channels

AI Runtime caps job environment variables at 20 entries and 512 characters each, so configuration
is split:

- **Small scalars → environment variables** (`variables.yaml` → `databricks.yml`): only the values
  `command.sh` reads — `TRAINING_SCRIPT`, `RAY_HEAD_PORT`, `RAY_JOIN_RETRIES`,
  `RAY_JOIN_SLEEP_SECONDS`.
- **Training config + dataset → JSON files** in the code artifact: `src/training_config.json` and
  `src/training_examples.json`, read from disk by the driver.

**To retune training or change the data, edit the JSON files in `src/`** — no `databricks.yml` or
Python change needed.

`NODE_RANK`, `MASTER_ADDR`, `MASTER_PORT`, `NUM_NODES`, `LOCAL_WORLD_SIZE`, `WORLD_SIZE`, and
`CODE_SOURCE_PATH` are injected by AI Runtime at runtime; you do not set them.

## Deploy and run

```bash
databricks bundle validate --target prod --profile <PROFILE>
databricks bundle deploy  --target prod --profile <PROFILE>
databricks bundle run llama_finetuning --target prod --profile <PROFILE>   # resource key, not display name
```

Scale GPUs without editing files (multiple of 8; 8 GPUs per node):

```bash
databricks bundle run llama_finetuning --target prod --profile <PROFILE> --var="accelerator_count=16"
```

The job ships with its schedule **paused** (`schedule_pause_status: PAUSED`). Set it to `UNPAUSED`
in `variables.yaml` and redeploy to enable the cron.

## Environments

Three targets — **`prod`** (default), **`stage`**, and **`test`** — share one workspace host and
are isolated by `workspace.root_path` and a distinct `run_as` service principal each. Switch with
`--target`:

```bash
databricks bundle deploy --target stage --profile <PROFILE>
```

`stage` and `test` ship with placeholder service principal ids (`REPLACE_WITH_*`); set real ones
before deploying those environments.

## Reading the logs

A failed `ai_runtime_task` reports only a generic `INTERNAL_ERROR` in the Jobs UI and API — the
real traceback is **not** shown there. The per-node stdout/stderr is captured as **MLflow run
artifacts**:

```bash
databricks api get "/api/2.0/mlflow/artifacts/list?run_id=<RUN_ID>&path=logs/node_0" --profile <PROFILE>
databricks api get "/api/2.0/mlflow/get-artifact?run_uuid=<RUN_ID>&path=logs/node_0/logs-0.chunk.txt" --profile <PROFILE>
```

`node_0` is the driver and is the most useful; other nodes are `node_1`, `node_2`, and so on.

## Checkpointing and resume

Checkpoints go to the UC Volume at `ray_storage_path`, under `ray_results/<ray_train_run_name>/`
(`checkpoints_to_keep` bounds retention). A rerun against the same run name **auto-resumes** from
the latest checkpoint; for a fresh run, use a new `ray_train_run_name`. To resume from a specific
checkpoint, set `resume_checkpoint` to the checkpoint **directory** (the folder that contains
`checkpoint.ckpt`) and use a new `ray_train_run_name`. Resume restores the epoch counter and trains
only the remaining epochs up to `max_epochs`.

## Clean up

```bash
databricks bundle destroy --target prod --profile <PROFILE>
```

This removes the job and the deployed bundle files for that environment. Checkpoints already
written to the UC Volume are not deleted.
