# AIR_NOTEBOOK — run the training in a Databricks notebook on serverless GPU

`train_llama_fsdp_notebook.py` is a **self-contained Databricks notebook** that runs the same
TinyLlama **Ray Train + Lightning FSDP** training as `src/train_llama_ray_fsdp.py`, but
interactively on **serverless GPU** compute instead of the DAB `ai_runtime_task` job. All of the
same metrics are logged to an MLflow run.

## How to run

1. Import `train_llama_fsdp_notebook.py` into your workspace (Workspace → Import, or `databricks
   workspace import`). It is a Databricks notebook source file.
2. Attach it to **serverless GPU** compute.
3. Set two values in the **Configuration** cell:
   - `RAY_STORAGE_PATH` — a UC Volume path you can write to (for Ray Train checkpoints).
   - `EXPERIMENT_PATH` — leave `None` to default to `/Users/<you>/air-ray-fsdp-llama-notebook`,
     or set an explicit experiment path.
4. **Run all.** The run id is printed; open it from the **Experiments** page (or the notebook's
   MLflow sidebar) and look at the **Metrics** tab.

## How it differs from the DAB / `src/` path

| | DAB `ai_runtime_task` (`src/`) | This notebook |
|---|---|---|
| Cluster bootstrap | `command.sh` starts a Ray head on node 0, workers join | starts a **local Ray cluster** (`ray.init()`) on the attached GPU node |
| Scale | multi-node (`accelerator_count` GPUs across nodes) | **single-node, all GPUs on the node** (`num_workers = torch.cuda.device_count()`) |
| Config + dataset | `training_config.json` + `training_examples.json` in the code tgz | **inlined** in the notebook |
| MLflow run | created by AI Runtime; driver logs into it | the **notebook creates** the run via `mlflow.start_run()` and sets the experiment |
| Worker MLflow auth | provided by AI Runtime | the notebook captures its host + API token and passes them to the workers |

The training itself — `TorchTrainer` → `train_func` per GPU → Lightning `RayFSDPStrategy` wrapping
`LlamaDecoderLayer`, same hyperparameters, same `MLflowMetricsCallback` — is identical.

## Metrics logged (same as the job)

Per step: `train_loss_step`, `learning_rate`. Per epoch: `epoch`, `global_step`,
`train_loss_epoch`, `epoch_duration_seconds`, `samples_per_second`, `peak_gpu_memory_mb`.
Run-level: `total_training_seconds` + params (`model_name`, `max_epochs`, `learning_rate`,
`per_device_batch_size`, `world_size`, `fsdp_sharding_strategy`, `num_training_examples`). If
`mlflow.enable_system_metrics_logging()` succeeds, `system/*` GPU/CPU/mem metrics are captured too.

## Notes

- **Dataset:** the 24-example base set is inlined for readability. To reproduce the 240-example
  prod dataset, paste the full list from `src/training_examples.json` into `TRAINING_EXAMPLES`
  (or load it from a Volume).
- **Steps per epoch** = `ceil( ceil(num_examples / num_gpus) / per_device_batch_size )` — more
  GPUs shard the data further, so the per-epoch step count drops.
- **Multi-node:** this notebook is single-node multi-GPU. True multi-node training is what the
  DAB `ai_runtime_task` / `air` CLI path provides.
