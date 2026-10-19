# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A Databricks Asset Bundle (DAB) that fine-tunes TinyLlama across multiple GPU nodes on
Databricks **AI Runtime** (serverless GPU), using **Ray Train** to launch one worker per GPU
and PyTorch **Lightning FSDP** (`RayFSDPStrategy`) for the sharded distributed training. The
job is a single `ai_runtime_task` (Public Preview) that provisions `GPU_8xH100` nodes, runs
`command.sh` on every node, and trains via `src/train_llama_ray_fsdp.py`.

## Commands

Authenticate (the deploying identity; the job itself runs as the service principal via
`run_as`):

```bash
databricks auth describe --profile <PROFILE>
```

Bundle lifecycle (**requires Databricks CLI ≥ 1.20.0** — older CLIs don't know the
`ai_runtime_task` / `environment_variables` fields and silently drop them). There are three
targets — `prod` (default), `stage`, `test` — on the **same workspace host**, isolated by
`workspace.root_path` (`/Workspace/prod`, `/Workspace/stage`, `/Workspace/test`) and a
distinct `run_as` service principal each (see `variables.yaml` `targets:`). Swap `--target`
below for `stage`/`test`:

```bash
databricks bundle validate --target prod --profile <PROFILE>
databricks bundle deploy  --target prod --profile <PROFILE>
databricks bundle run llama_finetuning --target prod --profile <PROFILE>   # resource key, not display name
databricks bundle destroy --target prod --profile <PROFILE>
# scale without editing files: --var="accelerator_count=16"
#   GPUs-per-node depends on accelerator_type: GPU_8xH100 = 8/node (use a multiple of 8),
#   GPU_1xA10 = 1/node (any count). stage overrides to GPU_1xA10 x3 in variables.yaml.
```

No local tests/lint. `train_llama_ray_fsdp.py` cannot run outside AI Runtime (needs Ray +
GPUs). Local checks: `python3 -I -m py_compile src/train_llama_ray_fsdp.py` and
`databricks bundle validate`.

## Reading the logs (critical — the job hides them by default)

A failed `ai_runtime_task` reports only a generic `INTERNAL_ERROR` via the Jobs API, `air
logs`, and the run page's output tab — **no traceback anywhere obvious**. The real per-node
stdout/stderr is captured as **MLflow run artifacts** (tag `mlflow.databricks.airEnableLogsV2`).
To read them:

```bash
# 1. find the experiment. The DAB job sets mlflow_experiment_directory to the deploy root,
#    so it is <workspace_root_path>/air-ray-fsdp-llama (e.g. /Workspace/stage/air-ray-fsdp-llama).
#    A user/air run with no directory defaults to /Users/<deployer>@databricks.com/air-ray-fsdp-llama.
# 2. search runs -> get the MLflow run_id for the failed job run
databricks api post /api/2.0/mlflow/runs/search --json '{"experiment_ids":["<EXPID>"],"max_results":10}' --profile <P>
# 3. list and read the per-node log artifact (DBFS root reads are disabled; use get-artifact):
databricks api get "/api/2.0/mlflow/artifacts/list?run_id=<RID>&path=logs/node_0" --profile <P>
databricks api get "/api/2.0/mlflow/get-artifact?run_uuid=<RID>&path=logs/node_0/logs-0.chunk.txt" --profile <P>
```

Alternative for fast iteration: the **`air` CLI** (see the `AIR_CLI/` section below) submits an
equivalent `ai_runtime_task` payload but runs as **you** and **streams logs live** (`databricks
air logs <RUN_ID>` — no MLflow hunt). Use it to reproduce/diagnose (it was how every real error
here was first seen). It is **not** a substitute for the DAB job — this project must deploy via
DAB env-to-env.

## The `AIR_CLI/` sibling runner (and the src/ ↔ AIR_CLI drift)

`AIR_CLI/` is a self-contained copy of the workload that runs through `databricks air run`
instead of DABs — a *user* run that streams logs live, for fast iteration/debugging:

```bash
cd AIR_CLI
databricks air run --file air_run.yaml --dry-run --profile <P>   # validate, no submit
databricks air run --file air_run.yaml --watch   --profile <P>   # submit + stream logs
databricks air run --file air_run.yaml --watch --override compute.num_accelerators=8 --profile <P>
databricks air list|logs|get|cancel <RUN_ID> --profile <P>
```

`air_run.yaml` mirrors the DAB `ai_runtime_task` field-for-field (see the mapping table in
`AIR_CLI/README.md`) and deliberately omits `mlflow_experiment_directory` /
`mlflow_artifact_location` for the same reasons as the DAB job (invariants 1 and 10).

**IMPORTANT — the `AIR_CLI/` copies have diverged from `src/`; they are NOT byte-identical**
(despite what `AIR_CLI/README.md` still claims). As of now:

- `command.sh` and `training_examples.json` — still identical to the DAB versions.
- `AIR_CLI/train_llama_ray_fsdp.py` — **ahead of `src/`**: it adds a seeded train/val split
  (`split_train_val`), a `validation_step`, a held-out val `DataLoader`, and per-step/per-epoch
  train+val loss logging. The DAB `src/` driver has none of this (train-loss only).
- `AIR_CLI/training_config.json` — adds the keys that driver reads (`validation_split`,
  `val_per_device_batch_size`, `num_sanity_val_steps`) and carries a **real** `ray_storage_path`
  Volume + a distinct `ray_train_run_name`; `src/training_config.json` keeps placeholder
  `<catalog>/<schema>/...` paths.
- `AIR_CLI/air_run.yaml` defaults to `GPU_1xA10` × 4 (H100 lines commented out).

So **a change in one tree is NOT reflected in the other** — edit both deliberately, or port the
validation logic into `src/` if the DAB job should gain it. Do not assume editing `src/` updates
the `air` runner (or vice versa).

## Configuration has two channels

AI Runtime caps `environment_variables` at **20 entries and 512 chars per value**, so config
is split:

1. **Small scalars → env vars** (declared in `variables.yaml`, mapped in `databricks.yml`'s
   `environment_variables` block). Only what `command.sh` reads: `TRAINING_SCRIPT`,
   `RAY_HEAD_PORT`, `RAY_JOIN_RETRIES`, `RAY_JOIN_SLEEP_SECONDS`.
2. **Training config + dataset → JSON files in the code artifact**:
   - `src/training_config.json` — every per-field training setting the driver consumes
     (model, FSDP dtypes/sharding, epochs, LR, cache env, Ray run settings, `resume_checkpoint`).
     Real JSON types (numbers/booleans).
   - `src/training_examples.json` — the instruction-tuning dataset (JSON list).

   `train_llama_ray_fsdp.py` reads both from **its own directory** (`SCRIPT_DIR =
   Path(__file__).resolve().parent`) via `_load_json_file`, then pulls typed values through
   `cfg` / `cfg_bool` / `cfg_dtype`. **Not** from env vars.

**To retune training / change data:** edit the JSON files in `src/` — no `databricks.yml` or
Python change. A missing key raises `Required key is missing from training_config.json`.

`NODE_RANK`, `MASTER_ADDR`, `MASTER_PORT`, `NUM_NODES`, `LOCAL_WORLD_SIZE`, `WORLD_SIZE` are
**AI Runtime-provided** at runtime (confirmed via env dump); `command.sh`/the driver read them
directly and they never appear in `variables.yaml`.

## Invariants — do NOT regress these (each was a multi-hour bug)

1. **`ai_runtime_task`: set `experiment` + `mlflow_run` + `mlflow_experiment_directory`;
   NEVER set `mlflow_artifact_location`.** Two separate hard-won points:
   - `mlflow_artifact_location` on a `dbfs:/Volumes/...` path makes AI Runtime fail at **task
     init, before `command.sh`**, with zero logs. Do not add it.
   - `mlflow_experiment_directory` **must** point at a folder the **`run_as` SP can write**
     (we use the deploy root `${var.workspace_root_path}`, which the top-level `permissions`
     block grants the SP `CAN_MANAGE`). If omitted, the experiment defaults to the *deployer's*
     `/Users/...` folder; a service-principal `run_as` has no access there, so AI Runtime's
     MLflow-run creation fails and the task dies with the **same opaque `INTERNAL_ERROR` and no
     logs** — a different cause from the Volume path (invariant 10), with identical symptoms.
     (A *user* `run_as` / the `air` CLI doesn't need this — it owns its default experiment.)
2. **`TRAINING_SCRIPT = train_llama_ray_fsdp.py`** (no `src/` prefix). AI Runtime sets
   `CODE_SOURCE_PATH` to the extracted code-source folder, which is the tgz's single
   top-level dir — i.e. `.../code_source/src`. `command.sh` does `cd "$CODE_SOURCE_PATH"`, so
   the script sits directly there. A `src/` prefix produces `src/src/...` file-not-found.
3. **Gradient clipping goes through FSDP.** `TinyLlamaLightningModule.configure_gradient_clipping`
   calls `self.trainer.model.clip_grad_norm_(...)`. Lightning's default norm clipping raises
   `MisconfigurationException: gradient_clip_algorithm='norm' is currently not supported for
   FSDPPrecision`. Do not reintroduce reliance on `Trainer(gradient_clip_val=...)` default clipping.
4. **The `run_as` service principal needs `WRITE_VOLUME`** (not just `READ_VOLUME`) on the
   checkpoint Volume (the `ray_storage_path` Volume configured in `training_config.json`), or the driver
   can't write checkpoints. Also, the deploying identity needs the `servicePrincipal.user`
   role on the SP to bind `run_as` at deploy time.
5. **`bundle.name` and `workspace.host` are pinned literals** in `databricks.yml` — DABs
   forbids variable interpolation for the auth host, and parameterizing the bundle name breaks
   the state path. The `bundle_name`/`workspace_host` variables are reference-only.
6. **CLI ≥ 1.20.0** (see Commands).
7. **`environment_variables` are applied only for a *user* `run_as`, NOT a service
   principal.** The launcher logs `Applying 0 user env var(s)` under SP `run_as`, so
   `RAY_HEAD_PORT`/`TRAINING_SCRIPT`/etc. arrive unset and `command.sh` (which uses `set -u`)
   dies with `unbound variable`. `command.sh` therefore self-defaults every value it needs
   via `${VAR:-default}`; do not make it hard-depend on the job's `environment_variables`.
   (AI-Runtime-injected vars — `NODE_RANK`, `MASTER_ADDR`, `LOCAL_WORLD_SIZE`,
   `CODE_SOURCE_PATH` — *are* always present.)
8. **Worker nodes must not use `ray start --block`.** It did not return after node 0 stopped
   the head, leaving the whole task hung in teardown until the 2-hour timeout (even though
   training finished and node 0 exited 0). Workers instead start Ray non-blocking, then poll
   `ray status` and exit once the head goes away (with a consecutive-failure guard so a
   transient blip doesn't drop a worker mid-training). Single-node runs are unaffected.
9. **Do not use `TorchTrainer(resume_from_checkpoint=…)`.** Ray Train V2 (ray ≥ ~2.5) removed
   it and *raises* a `DeprecationWarning` that crashes the run. Explicit, user-specified
   resume (`resume_checkpoint` in `training_config.json`) is handled inside `train_func` by
   passing `ckpt_path=<dir>/checkpoint.ckpt` to `lightning_trainer.fit()` — the same path used
   for automatic (`train.get_checkpoint()`) resume. Resuming restores the epoch counter, so it
   trains only the remaining epochs up to `max_epochs`; point `resume_checkpoint` at the
   checkpoint **directory** (contains `checkpoint.ckpt`) and use a fresh `ray_train_run_name`
   so the explicit checkpoint isn't overridden by auto-resume of a newer one in the same folder.
10. **A top-level `permissions` block must grant the `run_as` SP access to the deploy folder.**
    Files under `root_path` are owned by the *deploying user*, so when the job runs **as the
    service principal** it cannot read `command.sh` unless the SP has access to the bundle
    folder. Symptom: the task fails with `User lacks read access to file:
    <root_path>/files/command.sh` (run shows `Blocked` then `INTERNAL_ERROR`). The bundle sets
    a top-level `permissions:` entry granting the SP (`${var.service_principal_application_id}`)
    `CAN_MANAGE`; DABs applies it to the `root_path` folder and `files/`/`artifacts/` inherit it.
    A second entry grants the deployer (`${workspace.current_user.userName}`) `CAN_MANAGE` so
    future deploys keep access. This was only unnecessary under the old world-readable
    `/Shared/.bundle/...` root; do **not** drop the block now that `root_path` is `/Workspace/...`.

## Runtime flow (`bundle run`)

1. AI Runtime provisions nodes from `accelerator_type`/`accelerator_count` (`GPU_8xH100`, 8
   GPUs/node → `accelerator_count/8` nodes), builds the env from the `environments` spec
   (`base_environment: workspace-base-environments/databricks_ai_v6` + the pip deps), then
   runs `command.sh` on **every** node.
2. `command.sh` `cd "$CODE_SOURCE_PATH"` then branches on `NODE_RANK`: node 0 starts the Ray
   head and runs the driver (then its `EXIT` trap runs `ray stop`); other nodes join the head
   (retrying up to `RAY_JOIN_RETRIES`) and then poll `ray status`, exiting once node 0 stops
   the head (see invariant 8 — not `--block`).
3. The driver (`main()`) does `ray.init(address="auto")`, blocks in `wait_for_all_gpus` until
   `NUM_NODES × LOCAL_WORLD_SIZE` GPUs join, then launches `TorchTrainer`.
4. `TorchTrainer` runs `train_func` once per Ray worker (one GPU each); each builds a Lightning
   `Trainer(devices=1)` with `RayFSDPStrategy` wrapping `LlamaDecoderLayer`. Checkpoints report
   via `RayTrainReportCallback` to `RunConfig(storage_path=ray_storage_path)`.

## Checkpointing & resume

- Checkpoints go to the UC Volume at `ray_storage_path` (`ray_results/<ray_train_run_name>/`);
  `CheckpointConfig(num_to_keep=...)` bounds retention. There is no separate `checkpoint_dir`.
- Ray Train **auto-resumes** from the latest checkpoint if `storage_path/<run_name>` already
  has one (reruns against the same path resume rather than retrain). For a fresh run, use a new
  `ray_train_run_name` or clear that directory. Setting `resume_checkpoint` (a checkpoint
  **directory**) forces an explicit resume — see invariant 9 for the mechanism and the
  fresh-run-name requirement. Either way, resume restores the epoch counter and continues only
  the remaining epochs up to `max_epochs` (confirmed: resuming an epoch-2 checkpoint with
  `max_epochs=3` trained exactly one more epoch and wrote one new checkpoint).

## Deployment notes

- Ships with `schedule_pause_status: PAUSED`; set `UNPAUSED` in `variables.yaml` + redeploy to
  enable the cron.
- `artifacts.code` packages `src/` (script + both JSON files) into `./dist/code.tgz`; the
  `files:` entry naming that output is **required** for a `tgz` artifact. Keep `src/` free of
  stray files (e.g. `__pycache__`) — they ship too.
- Three targets (`prod`/`stage`/`test`) are defined in `databricks.yml` `targets:` with the
  per-target `workspace_root_path`, `command_path`, and `service_principal_application_id`
  set per target in `variables.yaml` `targets:` (no shared default). **`command_path` must track
  `root_path`**: it is
  `<root_path>/files/command.sh` (e.g. `/Workspace/stage/files/command.sh` for `/Workspace/stage`).
  If you change a target's `root_path`, change its `command_path` too, or AI Runtime can't find
  `command.sh`. All three targets currently carry concrete SP application ids in `variables.yaml`
  (the `test` id is still flagged with a `TODO` to confirm). Verify each target's
  `service_principal_application_id` is the right SP for *your* workspace — and keep the top-level
  `permissions:` SP reference in sync with it — before deploying that target.
- Before first deploy to a new workspace, set: `workspace.host` (literal in `databricks.yml`),
  per-target `service_principal_application_id` / `workspace_root_path` / `command_path` (in
  `variables.yaml`), and the Volume paths in `src/training_config.json` (`ray_storage_path`).
  Grant each target's SP `WRITE_VOLUME` on the checkpoint Volume, and the deploying identity the
  `servicePrincipal.user` role on that SP.
- The top-level `permissions:` block grants the `run_as` SP (and the deployer) `CAN_MANAGE` so
  the SP can read the deployed code folder — required for any non-shared `root_path` (see
  invariant 10). Keep its SP reference in sync with the per-target `service_principal_application_id`.
- `README.md` is the public overview + quickstart, kept in sync with this design; this `CLAUDE.md`
  is the authoritative engineering reference (architecture + invariants). A customer-facing
  deployment runbook is maintained separately (Google Doc).
