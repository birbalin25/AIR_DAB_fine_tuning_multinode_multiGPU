# Persistent AI Runtime Ray + Lightning FSDP job

This bundle uses the attached TinyLlama Lightning notebook as the training implementation. The notebook's inline instruction examples, TinyLlama model, tokenizer formatting, BF16 model loading, FSDP decoder-layer wrapping, activation checkpointing, optimizer settings, gradient clipping, and checkpoint resume behavior are preserved.

The notebook-only `ddp_notebook` strategy and the later single-GPU fallback trainer are intentionally removed. In the persistent job, AI Runtime provisions the nodes, Ray Train launches one worker per GPU, and `RayFSDPStrategy` integrates Lightning FSDP with Ray Train.

## Files

```text
air_ray_fsdp_llama_dabs/
├── databricks.yml
├── variables.yaml
├── command.sh
└── src/
    └── train_llama_ray_fsdp.py
```

## Parameterization

All user-configurable values from the attached notebook are declared in `variables.yaml`, including:

* TinyLlama model name and checkpoint paths.
* Epochs, batch size, learning rate, weight decay, sequence length, workers, logging, and gradient clipping.
* FSDP sharding, dtypes, activation checkpointing, CPU offload, and Lightning precision.
* Hugging Face cache and CUDA allocator settings.
* The inline training examples as `inline_training_examples_json`.
* AI Runtime, Ray, MLflow, schedule, workspace, and service-principal settings.

The source file reads configuration only from environment variables injected by the DAB job. `NODE_RANK`, `MASTER_ADDR`, `NUM_NODES`, and `LOCAL_WORLD_SIZE` remain AI Runtime-provided runtime values.

## Preparation

1. Edit `variables.yaml`:
   * `workspace_host`
   * `service_principal_application_id`
   * `workspace_root_path`
   * `checkpoint_dir`
   * `ray_storage_path`
   * MLflow and Unity Catalog Volume paths
2. Ensure the service principal can deploy and run the bundle.
3. Grant the service principal write access to the configured checkpoint and MLflow Volume paths.
4. Use Databricks CLI 1.20.0 or later.

## Service-principal authentication

Do not store the OAuth secret in the bundle:

```bash
export DATABRICKS_HOST="https://your-workspace.cloud.databricks.com"
export DATABRICKS_CLIENT_ID="<service-principal-application-id>"
export DATABRICKS_CLIENT_SECRET="<oauth-client-secret>"

databricks auth describe
```

## Validate, deploy, and run

```bash
cd air_ray_fsdp_llama_dabs

databricks bundle validate --target prod
databricks bundle deploy --target prod
databricks bundle run llama_finetuning --target prod
```

The job is persistent and starts with its schedule paused. To enable the schedule, change `schedule_pause_status` to `UNPAUSED` and redeploy:

```bash
databricks bundle deploy --target prod
```

## Resume from a checkpoint

Set `resume_checkpoint` to a directory containing the reported AIR checkpoint, including `checkpoint.ckpt`:

```yaml
resume_checkpoint:
  default: /Volumes/main/default/air_checkpoints/notebook_tinyllama_fsdp/<checkpoint-directory>
```

The job also resumes automatically from the latest Ray AIR checkpoint after a retry when `resume_checkpoint` is empty.

## Destroy

```bash
databricks bundle destroy --target prod
```
