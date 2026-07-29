from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_config(path: str, override_revision: str | None) -> dict[str, Any]:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if override_revision:
        config["model_revision"] = override_revision
    if config.get("model_revision") in {None, "", "REQUIRED_COMMIT_SHA"}:
        raise ValueError("pass --model-revision with an immutable commit SHA")
    return config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--model-revision")
    args = parser.parse_args()
    config = _load_config(args.config, args.model_revision)
    try:
        import torch
        from datasets import load_dataset
        from peft import LoraConfig
        from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
        from trl import SFTConfig, SFTTrainer
    except ImportError as exc:
        raise RuntimeError("install requirements-ml.txt before training") from exc
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if config["bf16"] else torch.float16,
    )
    model = AutoModelForImageTextToText.from_pretrained(
        config["model_id"], revision=config["model_revision"], quantization_config=quantization
    )
    processor = AutoProcessor.from_pretrained(config["model_id"], revision=config["model_revision"])
    dataset = load_dataset("json", data_files=config["dataset_path"], split="train")
    peft_config = LoraConfig(
        r=config["lora_r"],
        lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"],
        target_modules=config["target_modules"],
        task_type="CAUSAL_LM",
    )
    training_args = SFTConfig(
        output_dir=config["output_dir"],
        seed=config["seed"],
        learning_rate=config["learning_rate"],
        num_train_epochs=config["num_train_epochs"],
        per_device_train_batch_size=config["per_device_train_batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        gradient_checkpointing=config["gradient_checkpointing"],
        bf16=config["bf16"],
        max_length=config["max_length"],
        report_to="none",
    )
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=processor,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(config["output_dir"])
    processor.save_pretrained(config["output_dir"])


if __name__ == "__main__":
    main()

