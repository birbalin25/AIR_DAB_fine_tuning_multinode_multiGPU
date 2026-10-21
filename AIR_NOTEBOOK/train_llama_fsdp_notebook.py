# Databricks notebook source
# MAGIC %md
# MAGIC # Fine-tune TinyLlama with Ray Train + Lightning FSDP on Serverless GPU
# MAGIC
# MAGIC Self-contained notebook port of `src/train_llama_ray_fsdp.py`. It runs the **same**
# MAGIC training (Ray Train `TorchTrainer` + PyTorch Lightning `RayFSDPStrategy`) and logs the
# MAGIC **same metrics** to MLflow, but is adapted for an interactive **serverless GPU** notebook:
# MAGIC
# MAGIC - **No `command.sh` / AI Runtime bootstrap.** This notebook starts a **local Ray cluster**
# MAGIC   on the attached GPU compute and scales to all GPUs on the node
# MAGIC   (`num_workers = torch.cuda.device_count()`), one Lightning worker per GPU.
# MAGIC - **Config + dataset are inlined** (no `training_config.json` / `training_examples.json`).
# MAGIC - **The notebook creates the MLflow run itself** and sets the experiment (the DAB relied on
# MAGIC   AI Runtime to create the run). Metrics are logged into that run so they appear in its
# MAGIC   **Metrics** tab.
# MAGIC
# MAGIC > Scope: this runs **single-node, multi-GPU** (all GPUs on the serverless GPU node).
# MAGIC > True multi-node is what the DAB `ai_runtime_task` / `air` CLI path is for.
# MAGIC
# MAGIC **How to run:** attach to **serverless GPU** compute, then *Run all*.

# COMMAND ----------

# MAGIC %md ## 0. Install dependencies (match the DAB environment), then restart Python

# COMMAND ----------

# MAGIC %pip install -q "lightning>=2.4,<2.6" "transformers>=4.45,<5" "accelerate>=0.34" "safetensors>=0.4" "ray[train]>=2.40"

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Configuration + dataset (inline — self-contained)
# MAGIC
# MAGIC Same keys as `training_config.json`. Edit here to retune. `TRAINING_EXAMPLES` is the inline
# MAGIC instruction dataset — the 24-example base set is included for readability; paste the full
# MAGIC 240-example list from `src/training_examples.json` (or load it from a Volume) to reproduce
# MAGIC the prod dataset.

# COMMAND ----------

# ---- Notebook-specific settings ----
# Where to log metrics. None -> /Users/<you>/air-ray-fsdp-llama-notebook (you can write there).
EXPERIMENT_PATH = None
# A UC Volume path Ray Train writes checkpoints to. Change to a Volume you can write to.
RAY_STORAGE_PATH = "/Volumes/serverless_stable_14ey07_catalog/air/air_checkpoints/notebook_results"

# ---- Training config (mirrors src/training_config.json) ----
CONFIG = {
    "model_name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "resume_checkpoint": "",
    "random_seed": 42,
    "max_epochs": 3,
    "per_device_batch_size": 2,
    "learning_rate": 0.00002,
    "weight_decay": 0.01,
    "max_length": 256,
    "dataloader_num_workers": 2,
    "log_every_n_steps": 1,
    "mlflow_log_metrics": True,
    "gradient_clip_val": 1.0,
    "lightning_precision": "32-true",
    "model_torch_dtype": "bfloat16",
    "fsdp_sharding_strategy": "FULL_SHARD",
    "fsdp_param_dtype": "bfloat16",
    "fsdp_reduce_dtype": "float32",
    "fsdp_buffer_dtype": "float32",
    "fsdp_use_activation_checkpointing": True,
    "fsdp_use_cpu_offload": False,
    "fsdp_cast_forward_inputs": True,
    "float32_matmul_precision": "high",
    "hf_home": "/tmp/huggingface",
    "transformers_cache": "/tmp/huggingface/transformers",
    "tokenizers_parallelism": "false",
    "pytorch_cuda_alloc_conf": "expandable_segments:True",
    "hf_hub_disable_xet": "true",
    "checkpoints_to_keep": 3,
    "ray_storage_path": RAY_STORAGE_PATH,
    "ray_train_run_name": "tinyllama-fsdp-notebook-1",
    "ray_max_failures": 1,
}

TRAINING_EXAMPLES = [
    {"instruction": "What is Databricks?", "input": "", "output": "Databricks is a data and AI platform for analytics, machine learning, and generative AI."},
    {"instruction": "Explain a Python list.", "input": "", "output": "A Python list is an ordered, mutable collection that can contain multiple values."},
    {"instruction": "What does SQL SELECT do?", "input": "", "output": "SELECT reads columns and rows from one or more tables or views."},
    {"instruction": "Define supervised learning.", "input": "", "output": "Supervised learning trains a model from examples that contain inputs and known target labels."},
    {"instruction": "What is a GPU useful for?", "input": "", "output": "A GPU performs many parallel numerical operations and is useful for deep learning workloads."},
    {"instruction": "Write a concise greeting.", "input": "", "output": "Hello, it is good to meet you."},
    {"instruction": "Convert the number 5 to a string in Python.", "input": "", "output": "Use str(5), which returns the string '5'."},
    {"instruction": "What is JSON?", "input": "", "output": "JSON is a text format commonly used to represent structured data with objects and arrays."},
    {"instruction": "Explain model fine-tuning.", "input": "", "output": "Fine-tuning continues training a pretrained model on a narrower dataset for a specific task or style."},
    {"instruction": "What is a database table?", "input": "", "output": "A table stores related records in rows, with each column representing an attribute."},
    {"instruction": "What does an optimizer do in neural network training?", "input": "", "output": "An optimizer updates model parameters using gradients to reduce the training loss."},
    {"instruction": "Explain batch size.", "input": "", "output": "Batch size is the number of examples processed together before an optimizer update."},
    {"instruction": "What is an epoch?", "input": "", "output": "An epoch is one complete pass through the training dataset."},
    {"instruction": "What does FSDP mean?", "input": "", "output": "FSDP means Fully Sharded Data Parallel; it shards model parameters, gradients, and optimizer state across GPUs."},
    {"instruction": "Give one benefit of checkpointing.", "input": "", "output": "Checkpointing allows training to resume after interruption without starting from the beginning."},
    {"instruction": "What is a tokenizer?", "input": "", "output": "A tokenizer converts text into token IDs that a language model can process."},
    {"instruction": "Explain learning rate.", "input": "", "output": "The learning rate controls the size of parameter updates during optimization."},
    {"instruction": "What is a training dataset?", "input": "", "output": "A training dataset contains examples used to learn model parameters."},
    {"instruction": "What does torch.cuda.is_available() return?", "input": "", "output": "It returns True when PyTorch can access a CUDA-capable GPU."},
    {"instruction": "Write a short definition of machine learning.", "input": "", "output": "Machine learning uses data to learn patterns that support predictions or decisions."},
    {"instruction": "Why use gradient clipping?", "input": "", "output": "Gradient clipping limits unusually large gradients and can improve training stability."},
    {"instruction": "What is an attention mechanism?", "input": "", "output": "Attention lets a model weigh the importance of different input tokens when computing a representation."},
    {"instruction": "What is a checkpoint file?", "input": "", "output": "A checkpoint file stores enough training state to resume or inspect a model run."},
    {"instruction": "What is the purpose of a validation set?", "input": "", "output": "A validation set estimates performance on held-out examples during model development."},
]

print(f"{len(TRAINING_EXAMPLES)} training examples; checkpoints -> {RAY_STORAGE_PATH}")

# COMMAND ----------

# MAGIC %md ## 2. Typed config accessors + derived constants (mirrors the driver)

# COMMAND ----------

import os
import time
from functools import partial

import torch


def cfg(key):
    if key not in CONFIG:
        raise KeyError(f"Required key is missing from CONFIG: {key}")
    return CONFIG[key]


def cfg_bool(key):
    value = cfg(key)
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def cfg_dtype(key):
    values = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    normalized = str(cfg(key)).lower()
    if normalized not in values:
        raise ValueError(f"Unsupported dtype for {key}: {normalized}")
    return values[normalized]


MODEL_NAME = str(cfg("model_name"))
RESUME_CHECKPOINT = str(CONFIG.get("resume_checkpoint", "")).strip()
RANDOM_SEED = int(cfg("random_seed"))
MAX_EPOCHS = int(cfg("max_epochs"))
PER_DEVICE_BATCH_SIZE = int(cfg("per_device_batch_size"))
LEARNING_RATE = float(cfg("learning_rate"))
WEIGHT_DECAY = float(cfg("weight_decay"))
MAX_LENGTH = int(cfg("max_length"))
DATALOADER_NUM_WORKERS = int(cfg("dataloader_num_workers"))
LOG_EVERY_N_STEPS = int(cfg("log_every_n_steps"))
GRADIENT_CLIP_VAL = float(cfg("gradient_clip_val"))
LIGHTNING_PRECISION = str(cfg("lightning_precision"))
MODEL_TORCH_DTYPE = cfg_dtype("model_torch_dtype")
FSDP_SHARDING_STRATEGY = str(cfg("fsdp_sharding_strategy"))
FSDP_PARAM_DTYPE = cfg_dtype("fsdp_param_dtype")
FSDP_REDUCE_DTYPE = cfg_dtype("fsdp_reduce_dtype")
FSDP_BUFFER_DTYPE = cfg_dtype("fsdp_buffer_dtype")
FSDP_USE_ACTIVATION_CHECKPOINTING = cfg_bool("fsdp_use_activation_checkpointing")
FSDP_USE_CPU_OFFLOAD = cfg_bool("fsdp_use_cpu_offload")
FSDP_CAST_FORWARD_INPUTS = cfg_bool("fsdp_cast_forward_inputs")
FLOAT32_MATMUL_PRECISION = str(cfg("float32_matmul_precision"))
RAY_TRAIN_RUN_NAME = str(cfg("ray_train_run_name"))
RAY_MAX_FAILURES = int(cfg("ray_max_failures"))
CHECKPOINTS_TO_KEEP = int(cfg("checkpoints_to_keep"))
MLFLOW_LOG_METRICS = cfg_bool("mlflow_log_metrics")

torch.set_float32_matmul_precision(FLOAT32_MATMUL_PRECISION)

# COMMAND ----------

# MAGIC %md ## 3. Dataset, model, callbacks, and the per-worker training function
# MAGIC
# MAGIC These are copied verbatim from `src/train_llama_ray_fsdp.py` (the `MLflowMetricsCallback`
# MAGIC logs every requested metric: per-step `train_loss`/`learning_rate`, and per-epoch `epoch`,
# MAGIC `global_step`, `train_loss`, `epoch_duration_seconds`, `samples_per_second`,
# MAGIC `peak_gpu_memory_mb`). `train_func` additionally sets Databricks auth env on each worker so
# MAGIC its best-effort MLflow logging can reach the tracking server.

# COMMAND ----------

import lightning.pytorch as pl
from ray import train
from ray.train.lightning import (
    RayFSDPStrategy,
    RayLightningEnvironment,
    RayTrainReportCallback,
    prepare_trainer,
)
from torch.distributed.fsdp import MixedPrecision
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama.modeling_llama import LlamaDecoderLayer


class InlineInstructionDataset(Dataset):
    def __init__(self, examples, tokenizer, max_length):
        self.rows = []
        for example in examples:
            user_text = example["instruction"]
            if example["input"]:
                user_text = f"{user_text}\n\nAdditional context:\n{example['input']}"
            messages = [
                {"role": "user", "content": user_text},
                {"role": "assistant", "content": example["output"]},
            ]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            encoded = tokenizer(
                text,
                max_length=max_length,
                padding="max_length",
                truncation=True,
                return_attention_mask=True,
            )
            labels = [
                token_id if mask == 1 else -100
                for token_id, mask in zip(encoded["input_ids"], encoded["attention_mask"])
            ]
            self.rows.append(
                {
                    "input_ids": torch.tensor(encoded["input_ids"], dtype=torch.long),
                    "attention_mask": torch.tensor(encoded["attention_mask"], dtype=torch.long),
                    "labels": torch.tensor(labels, dtype=torch.long),
                }
            )

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


class TinyLlamaLightningModule(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, torch_dtype=MODEL_TORCH_DTYPE, low_cpu_mem_usage=True
        )
        self.model.config.use_cache = False
        self.model.gradient_checkpointing_enable()

    def forward(self, input_ids, attention_mask, labels=None):
        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    def training_step(self, batch, batch_idx):
        outputs = self(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        self.log(
            "train_loss",
            outputs.loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=batch["input_ids"].shape[0],
        )
        return outputs.loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    def configure_gradient_clipping(self, optimizer, gradient_clip_val=None, gradient_clip_algorithm=None):
        # FSDPPrecision does not support Lightning's default norm clipping; clip via the
        # FSDP-wrapped module's own sharding-aware clip_grad_norm_.
        if gradient_clip_val is None:
            return
        self.trainer.model.clip_grad_norm_(float(gradient_clip_val))


class ProgressCallback(pl.Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.is_global_zero:
            loss = trainer.callback_metrics.get("train_loss_epoch") or trainer.callback_metrics.get("train_loss")
            loss_value = float(loss.detach().cpu()) if loss is not None else float("nan")
            print(
                f"epoch={trainer.current_epoch + 1}/{trainer.max_epochs} "
                f"global_step={trainer.global_step} train_loss={loss_value:.4f}",
                flush=True,
            )


class MLflowMetricsCallback(pl.Callback):
    """Log training metrics into the MLflow run from global rank 0 (best-effort)."""

    def __init__(self, client, run_id, log_every_n_steps, world_size):
        self.client = client
        self.run_id = run_id
        self.log_every_n_steps = max(1, int(log_every_n_steps))
        self.world_size = max(1, int(world_size))
        self._epoch_start = None
        self._epoch_samples = 0
        self._warned = False

    def _log(self, key, value, step):
        if value is None:
            return
        try:
            self.client.log_metric(self.run_id, key, float(value), step=int(step))
        except Exception as error:  # never break training on a logging failure
            if not self._warned:
                print(f"MLflow metric logging failed ({key}): {error}", flush=True)
                self._warned = True

    @staticmethod
    def _metric(trainer, *names):
        for name in names:
            value = trainer.callback_metrics.get(name)
            if value is not None:
                return float(value.detach().cpu())
        return None

    def on_train_epoch_start(self, trainer, pl_module):
        self._epoch_start = time.time()
        self._epoch_samples = 0
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self._epoch_samples += int(batch["input_ids"].shape[0])
        if not trainer.is_global_zero:
            return
        step = trainer.global_step
        if step % self.log_every_n_steps != 0:
            return
        self._log("train_loss_step", self._metric(trainer, "train_loss_step", "train_loss"), step)
        if trainer.optimizers:
            self._log("learning_rate", trainer.optimizers[0].param_groups[0].get("lr"), step)

    def on_train_epoch_end(self, trainer, pl_module):
        if not trainer.is_global_zero:
            return
        step = trainer.global_step
        self._log("epoch", trainer.current_epoch + 1, step)
        self._log("global_step", step, step)
        self._log("train_loss_epoch", self._metric(trainer, "train_loss_epoch", "train_loss"), step)
        if self._epoch_start is not None:
            duration = time.time() - self._epoch_start
            self._log("epoch_duration_seconds", duration, step)
            if duration > 0:
                self._log("samples_per_second", (self._epoch_samples * self.world_size) / duration, step)
        if torch.cuda.is_available():
            self._log("peak_gpu_memory_mb", torch.cuda.max_memory_allocated() / (1024 * 1024), step)


def train_func(config):
    """Runs once per Ray worker, using one GPU per worker."""
    # Workers are separate processes; set HF caches and propagate Databricks auth so the
    # worker can reach the MLflow tracking server.
    os.environ.setdefault("TOKENIZERS_PARALLELISM", str(CONFIG["tokenizers_parallelism"]))
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", str(CONFIG["pytorch_cuda_alloc_conf"]))
    os.environ.setdefault("HF_HOME", str(CONFIG["hf_home"]))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(CONFIG["transformers_cache"]))
    os.environ.setdefault("HF_HUB_DISABLE_XET", str(CONFIG["hf_hub_disable_xet"]))
    if config.get("databricks_host"):
        os.environ["DATABRICKS_HOST"] = config["databricks_host"]
    if config.get("databricks_token"):
        os.environ["DATABRICKS_TOKEN"] = config["databricks_token"]

    rank = train.get_context().get_world_rank()
    pl.seed_everything(RANDOM_SEED, workers=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_dataset = InlineInstructionDataset(TRAINING_EXAMPLES, tokenizer, MAX_LENGTH)
    train_loader = DataLoader(
        train_dataset,
        batch_size=PER_DEVICE_BATCH_SIZE,
        shuffle=True,
        drop_last=False,
        num_workers=DATALOADER_NUM_WORKERS,
        pin_memory=True,
        persistent_workers=DATALOADER_NUM_WORKERS > 0,
    )

    auto_wrap_policy = partial(transformer_auto_wrap_policy, transformer_layer_cls={LlamaDecoderLayer})
    fsdp_strategy = RayFSDPStrategy(
        auto_wrap_policy=auto_wrap_policy,
        activation_checkpointing_policy=({LlamaDecoderLayer} if FSDP_USE_ACTIVATION_CHECKPOINTING else None),
        sharding_strategy=FSDP_SHARDING_STRATEGY,
        cpu_offload=FSDP_USE_CPU_OFFLOAD,
        mixed_precision=MixedPrecision(
            param_dtype=FSDP_PARAM_DTYPE,
            reduce_dtype=FSDP_REDUCE_DTYPE,
            buffer_dtype=FSDP_BUFFER_DTYPE,
            cast_forward_inputs=FSDP_CAST_FORWARD_INPUTS,
        ),
    )

    callbacks = [RayTrainReportCallback(), ProgressCallback()]
    mlflow_run_id = config.get("mlflow_run_id")
    mlflow_tracking_uri = config.get("mlflow_tracking_uri")
    if config.get("mlflow_log_metrics") and mlflow_run_id:
        try:
            import mlflow
            from mlflow.tracking import MlflowClient

            if mlflow_tracking_uri:
                mlflow.set_tracking_uri(mlflow_tracking_uri)
            client = MlflowClient(tracking_uri=mlflow_tracking_uri) if mlflow_tracking_uri else MlflowClient()
            callbacks.append(
                MLflowMetricsCallback(
                    client=client,
                    run_id=mlflow_run_id,
                    log_every_n_steps=LOG_EVERY_N_STEPS,
                    world_size=train.get_context().get_world_size(),
                )
            )
        except Exception as error:  # never let metric setup break training
            if rank == 0:
                print(f"MLflow metrics callback not attached: {error}", flush=True)

    lightning_trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        num_nodes=1,
        strategy=fsdp_strategy,
        precision=LIGHTNING_PRECISION,
        max_epochs=MAX_EPOCHS,
        gradient_clip_val=GRADIENT_CLIP_VAL,
        log_every_n_steps=LOG_EVERY_N_STEPS,
        plugins=[RayLightningEnvironment()],
        callbacks=callbacks,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        enable_checkpointing=False,
        use_distributed_sampler=True,
    )
    lightning_trainer = prepare_trainer(lightning_trainer)

    model = TinyLlamaLightningModule()
    ray_checkpoint = train.get_checkpoint()
    if ray_checkpoint:
        with ray_checkpoint.as_directory() as checkpoint_dir:
            checkpoint_path = os.path.join(checkpoint_dir, RayTrainReportCallback.CHECKPOINT_NAME)
            if rank == 0:
                print(f"Resuming from checkpoint: {checkpoint_path}", flush=True)
            lightning_trainer.fit(model=model, train_dataloaders=train_loader, ckpt_path=checkpoint_path)
    elif RESUME_CHECKPOINT:
        checkpoint_path = os.path.join(RESUME_CHECKPOINT, RayTrainReportCallback.CHECKPOINT_NAME)
        if rank == 0:
            print(f"Explicitly resuming from {checkpoint_path}", flush=True)
        lightning_trainer.fit(model=model, train_dataloaders=train_loader, ckpt_path=checkpoint_path)
    else:
        lightning_trainer.fit(model=model, train_dataloaders=train_loader)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Set the MLflow experiment and capture Databricks auth for the workers

# COMMAND ----------

import mlflow

# Resolve the experiment path (defaults to your /Users folder, which you can always write to).
if EXPERIMENT_PATH is None:
    try:
        _user = spark.sql("SELECT current_user()").collect()[0][0]
        EXPERIMENT_PATH = f"/Users/{_user}/air-ray-fsdp-llama-notebook"
    except Exception:
        EXPERIMENT_PATH = "/Shared/air-ray-fsdp-llama-notebook"

mlflow.set_experiment(EXPERIMENT_PATH)
TRACKING_URI = mlflow.get_tracking_uri()  # "databricks" in a Databricks notebook

# Optional: let MLflow capture per-node GPU/CPU/mem/disk system metrics too (best-effort).
try:
    mlflow.enable_system_metrics_logging()
except Exception as _e:
    print("System-metrics logging not enabled:", _e)

# The workers are separate processes and may not inherit the notebook's MLflow credentials,
# so capture the notebook's host + short-lived API token and hand them to each worker.
try:
    _ctx = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
    DB_HOST = _ctx.apiUrl().get()
    DB_TOKEN = _ctx.apiToken().get()
except Exception as _e:
    DB_HOST, DB_TOKEN = None, None
    print("Could not capture notebook auth; workers will rely on inherited credentials:", _e)

print(f"experiment={EXPERIMENT_PATH}  tracking_uri={TRACKING_URI}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Launch training on all GPUs of the node, logging into one MLflow run

# COMMAND ----------

import ray
from ray.train import CheckpointConfig, FailureConfig, RunConfig, ScalingConfig
from ray.train.torch import TorchTrainer
from mlflow.tracking import MlflowClient

num_gpus = torch.cuda.device_count()
if num_gpus == 0:
    raise RuntimeError("No GPU detected — attach this notebook to serverless GPU compute.")
print(f"Detected {num_gpus} GPU(s); using {num_gpus} Ray worker(s).")

if ray.is_initialized():
    ray.shutdown()
ray.init(ignore_reinit_error=True)  # local Ray cluster on this node

with mlflow.start_run(run_name=RAY_TRAIN_RUN_NAME) as run:
    run_id = run.info.run_id
    print(f"MLflow run_id={run_id}")

    trainer = TorchTrainer(
        train_loop_per_worker=train_func,
        train_loop_config={
            "mlflow_run_id": run_id,
            "mlflow_tracking_uri": TRACKING_URI,
            "mlflow_log_metrics": MLFLOW_LOG_METRICS,
            "databricks_host": DB_HOST,
            "databricks_token": DB_TOKEN,
        },
        scaling_config=ScalingConfig(num_workers=num_gpus, use_gpu=True),
        run_config=RunConfig(
            name=RAY_TRAIN_RUN_NAME,
            storage_path=RAY_STORAGE_PATH,
            checkpoint_config=CheckpointConfig(num_to_keep=CHECKPOINTS_TO_KEEP),
            failure_config=FailureConfig(max_failures=RAY_MAX_FAILURES),
        ),
    )

    train_start = time.time()
    result = trainer.fit()
    total_training_seconds = time.time() - train_start
    print(f"Training finished in {total_training_seconds:.1f}s: {result.metrics}")

    # Driver-side logging (the notebook process owns the run): run-level params + total time.
    if MLFLOW_LOG_METRICS:
        client = MlflowClient(tracking_uri=TRACKING_URI)
        try:
            client.log_metric(run_id, "total_training_seconds", float(total_training_seconds))
            run_params = {
                "model_name": MODEL_NAME,
                "max_epochs": MAX_EPOCHS,
                "learning_rate": LEARNING_RATE,
                "per_device_batch_size": PER_DEVICE_BATCH_SIZE,
                "world_size": num_gpus,
                "fsdp_sharding_strategy": FSDP_SHARDING_STRATEGY,
                "num_training_examples": len(TRAINING_EXAMPLES),
            }
            for _k, _v in run_params.items():
                try:
                    client.log_param(run_id, _k, _v)
                except Exception:
                    pass  # params are immutable; ignore if already set
        except Exception as _e:
            print("Driver-side MLflow logging failed:", _e)

    for _i, (checkpoint, metrics) in enumerate(result.best_checkpoints):
        print(f"retained_checkpoint[{_i}]={checkpoint} metrics={metrics}")

ray.shutdown()
print("Done.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. View the metrics
# MAGIC
# MAGIC Open the run above (printed `run_id`) in the experiment set at **`EXPERIMENT_PATH`**, via
# MAGIC the **Experiments** page or the notebook's MLflow sidebar. In the run's **Metrics** tab you
# MAGIC will see `train_loss_step`, `train_loss_epoch`, `epoch`, `global_step`,
# MAGIC `epoch_duration_seconds`, `samples_per_second`, `peak_gpu_memory_mb`, `learning_rate`, and
# MAGIC `total_training_seconds` (plus `system/*` metrics if system-metrics logging was enabled).
# MAGIC
# MAGIC Steps per epoch = `ceil( ceil(num_examples / num_gpus) / per_device_batch_size )`, so with
# MAGIC more GPUs the per-epoch step count drops (the data is sharded across workers).

# COMMAND ----------

displayHTML(
    f"<b>Experiment:</b> {EXPERIMENT_PATH}<br>"
    "Open <b>Experiments</b> in the left sidebar (or the flask icon on this notebook) to see the run and its Metrics tab."
)
