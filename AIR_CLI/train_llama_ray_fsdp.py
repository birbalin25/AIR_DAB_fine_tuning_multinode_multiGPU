#!/usr/bin/env python3
"""Ray Train + Lightning FSDP implementation based on the attached notebook.

The notebook's notebook-only ddp_notebook and single-GPU fallback are removed.
AI Runtime starts this driver on node 0, Ray launches one Lightning worker per
GPU, and RayFSDPStrategy provides the distributed FSDP integration.
"""

import json
import os
import time
from functools import partial
from pathlib import Path

import lightning.pytorch as pl
import torch
from ray import train
from ray.train import (
    CheckpointConfig,
    FailureConfig,
    RunConfig,
    ScalingConfig,
)
from ray.train.lightning import (
    RayFSDPStrategy,
    RayLightningEnvironment,
    RayTrainReportCallback,
    prepare_trainer,
)
from ray.train.torch import TorchTrainer
from torch.distributed.fsdp import MixedPrecision
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama.modeling_llama import LlamaDecoderLayer


def required_env(name: str) -> str:
    value = os.environ.get(name)
    if value is None or value == "":
        raise RuntimeError(f"Required environment variable is missing: {name}")
    return value


SCRIPT_DIR = Path(__file__).resolve().parent
TRAINING_CONFIG_FILE = "training_config.json"
TRAINING_EXAMPLES_FILE = "training_examples.json"


def _load_json_file(name: str):
    """Load a JSON data file shipped next to this script in the code artifact.

    Training config and the dataset travel as files (not environment variables)
    because the AI Runtime job caps each environment variable value at 512
    characters, which both blobs exceed.
    """
    path = SCRIPT_DIR / name
    if not path.is_file():
        raise FileNotFoundError(f"Could not find {name} at {path}")
    try:
        with path.open(encoding="utf-8") as handle:
            return json.load(handle)
    except json.JSONDecodeError as error:
        raise ValueError(f"{path} is not valid JSON: {error}") from error


def load_config() -> dict:
    config = _load_json_file(TRAINING_CONFIG_FILE)
    if not isinstance(config, dict):
        raise ValueError(f"{TRAINING_CONFIG_FILE} must contain a JSON object")
    return config


CONFIG = load_config()


def cfg(key: str):
    if key not in CONFIG:
        raise KeyError(f"Required key is missing from {TRAINING_CONFIG_FILE}: {key}")
    return CONFIG[key]


def cfg_bool(key: str) -> bool:
    value = cfg(key)
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def cfg_dtype(key: str) -> torch.dtype:
    values = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    normalized = str(cfg(key)).lower()
    if normalized not in values:
        raise ValueError(f"Unsupported dtype for {key}: {normalized}")
    return values[normalized]


def parse_examples():
    examples = _load_json_file(TRAINING_EXAMPLES_FILE)
    if not isinstance(examples, list) or not examples:
        raise ValueError(f"{TRAINING_EXAMPLES_FILE} must contain a non-empty JSON list")
    for index, example in enumerate(examples):
        if not all(key in example for key in ("instruction", "input", "output")):
            raise ValueError(f"Training example {index} must contain instruction, input, and output")
    return examples


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
RAY_STORAGE_PATH = str(cfg("ray_storage_path"))
RAY_CLUSTER_WAIT_SECONDS = int(cfg("ray_cluster_wait_seconds"))
RAY_CLUSTER_POLL_SECONDS = int(cfg("ray_cluster_poll_seconds"))
RAY_TRAIN_RUN_NAME = str(cfg("ray_train_run_name"))
RAY_MAX_FAILURES = int(cfg("ray_max_failures"))
CHECKPOINTS_TO_KEEP = int(cfg("checkpoints_to_keep"))
TRAINING_EXAMPLES = parse_examples()


os.environ.setdefault("TOKENIZERS_PARALLELISM", str(cfg("tokenizers_parallelism")))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", str(cfg("pytorch_cuda_alloc_conf")))
os.environ.setdefault("HF_HOME", str(cfg("hf_home")))
os.environ.setdefault("TRANSFORMERS_CACHE", str(cfg("transformers_cache")))
os.environ.setdefault("HF_HUB_DISABLE_XET", str(cfg("hf_hub_disable_xet")))

# Ray Train creates and writes checkpoints under RAY_STORAGE_PATH itself; the
# driver does not pre-create any Volume directory at import time.
torch.set_float32_matmul_precision(FLOAT32_MATMUL_PRECISION)


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
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
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
            MODEL_NAME,
            torch_dtype=MODEL_TORCH_DTYPE,
            low_cpu_mem_usage=True,
        )
        self.model.config.use_cache = False
        self.model.gradient_checkpointing_enable()

    def forward(self, input_ids, attention_mask, labels=None):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

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
        return torch.optim.AdamW(
            self.parameters(),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
        )

    def configure_gradient_clipping(
        self,
        optimizer,
        gradient_clip_val=None,
        gradient_clip_algorithm=None,
    ):
        # Lightning's FSDPPrecision does not support its default norm-based
        # gradient-clipping path (raises MisconfigurationException for
        # gradient_clip_algorithm='norm'). FSDP shards gradients across ranks, so
        # clip with the FSDP-wrapped module's own sharding-aware clip_grad_norm_.
        if gradient_clip_val is None:
            return
        self.trainer.model.clip_grad_norm_(float(gradient_clip_val))


class ProgressCallback(pl.Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.is_global_zero:
            loss = trainer.callback_metrics.get("train_loss")
            loss_value = float(loss.detach().cpu()) if loss is not None else float("nan")
            print(
                f"epoch={trainer.current_epoch + 1}/{trainer.max_epochs} "
                f"global_step={trainer.global_step} train_loss={loss_value:.4f}",
                flush=True,
            )


def train_func(config: dict):
    """Runs once per Ray worker, using one GPU per worker."""
    rank = train.get_context().get_world_rank()
    pl.seed_everything(RANDOM_SEED, workers=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_dataset = InlineInstructionDataset(
        examples=TRAINING_EXAMPLES,
        tokenizer=tokenizer,
        max_length=MAX_LENGTH,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=PER_DEVICE_BATCH_SIZE,
        shuffle=True,
        drop_last=False,
        num_workers=DATALOADER_NUM_WORKERS,
        pin_memory=True,
        persistent_workers=DATALOADER_NUM_WORKERS > 0,
    )

    auto_wrap_policy = partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={LlamaDecoderLayer},
    )

    fsdp_strategy = RayFSDPStrategy(
        auto_wrap_policy=auto_wrap_policy,
        activation_checkpointing_policy=(
            {LlamaDecoderLayer}
            if FSDP_USE_ACTIVATION_CHECKPOINTING
            else None
        ),
        sharding_strategy=FSDP_SHARDING_STRATEGY,
        cpu_offload=FSDP_USE_CPU_OFFLOAD,
        mixed_precision=MixedPrecision(
            param_dtype=FSDP_PARAM_DTYPE,
            reduce_dtype=FSDP_REDUCE_DTYPE,
            buffer_dtype=FSDP_BUFFER_DTYPE,
            cast_forward_inputs=FSDP_CAST_FORWARD_INPUTS,
        ),
    )

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
        callbacks=[
            RayTrainReportCallback(),
            ProgressCallback(),
        ],
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
        # Automatic resume (e.g. a Ray fault-tolerance retry): Ray hands us the checkpoint.
        with ray_checkpoint.as_directory() as checkpoint_dir:
            checkpoint_path = os.path.join(
                checkpoint_dir,
                RayTrainReportCallback.CHECKPOINT_NAME,
            )
            if rank == 0:
                print(f"Resuming from AIR checkpoint: {checkpoint_path}", flush=True)
            lightning_trainer.fit(
                model=model,
                train_dataloaders=train_loader,
                ckpt_path=checkpoint_path,
            )
    elif RESUME_CHECKPOINT:
        # Explicit, user-specified resume. Ray Train V2 removed TorchTrainer's
        # `resume_from_checkpoint` (it now raises a DeprecationWarning), so load the
        # checkpoint directly via Lightning's `ckpt_path` from the configured dir.
        checkpoint_path = os.path.join(
            RESUME_CHECKPOINT,
            RayTrainReportCallback.CHECKPOINT_NAME,
        )
        if rank == 0:
            print(f"Explicitly resuming from {checkpoint_path}", flush=True)
        lightning_trainer.fit(
            model=model,
            train_dataloaders=train_loader,
            ckpt_path=checkpoint_path,
        )
    else:
        lightning_trainer.fit(
            model=model,
            train_dataloaders=train_loader,
        )


def wait_for_all_gpus(ray_module):
    expected_nodes = int(required_env("NUM_NODES"))
    gpus_per_node = int(required_env("LOCAL_WORLD_SIZE"))
    expected_gpus = expected_nodes * gpus_per_node
    deadline = time.time() + RAY_CLUSTER_WAIT_SECONDS

    while time.time() < deadline:
        resources = ray_module.cluster_resources()
        available_gpus = int(resources.get("GPU", 0))
        alive_nodes = sum(1 for node in ray_module.nodes() if node.get("Alive"))
        print(
            f"Ray cluster status: {alive_nodes}/{expected_nodes} nodes, "
            f"{available_gpus}/{expected_gpus} GPUs",
            flush=True,
        )
        if available_gpus >= expected_gpus:
            return available_gpus
        time.sleep(RAY_CLUSTER_POLL_SECONDS)

    raise TimeoutError(
        f"Timed out waiting for {expected_nodes} nodes and {expected_gpus} GPUs"
    )


def main():
    import ray

    ray.init(address="auto")
    total_gpus = wait_for_all_gpus(ray)
    print(f"All Ray nodes joined; using {total_gpus} Ray workers", flush=True)

    trainer = TorchTrainer(
        train_loop_per_worker=train_func,
        scaling_config=ScalingConfig(
            num_workers=total_gpus,
            use_gpu=True,
        ),
        run_config=RunConfig(
            name=RAY_TRAIN_RUN_NAME,
            storage_path=RAY_STORAGE_PATH,
            checkpoint_config=CheckpointConfig(
                num_to_keep=CHECKPOINTS_TO_KEEP,
            ),
            failure_config=FailureConfig(
                max_failures=RAY_MAX_FAILURES,
            ),
        ),
    )

    result = trainer.fit()
    print(f"Training finished: {result.metrics}", flush=True)
    for index, (checkpoint, metrics) in enumerate(result.best_checkpoints):
        print(
            f"retained_checkpoint[{index}]={checkpoint} metrics={metrics}",
            flush=True,
        )
    ray.shutdown()


if __name__ == "__main__":
    main()
