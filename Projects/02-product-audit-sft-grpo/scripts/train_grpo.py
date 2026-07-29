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
    if not Path(config["sft_adapter_path"]).exists():
        raise FileNotFoundError("SFT adapter path does not exist")
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
        from peft import PeftModel
        from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
        from trl import GRPOConfig, GRPOTrainer
    except ImportError as exc:
        raise RuntimeError("install requirements-ml.txt before training") from exc
    from product_audit.rewards import product_audit_reward

    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if config["bf16"] else torch.float16,
    )
    base_model = AutoModelForImageTextToText.from_pretrained(
        config["model_id"], revision=config["model_revision"], quantization_config=quantization
    )
    model = PeftModel.from_pretrained(base_model, config["sft_adapter_path"], is_trainable=True)
    processor = AutoProcessor.from_pretrained(config["model_id"], revision=config["model_revision"])
    dataset = load_dataset("json", data_files=config["dataset_path"], split="train")
    training_args = GRPOConfig(
        output_dir=config["output_dir"],
        seed=config["seed"],
        learning_rate=config["learning_rate"],
        max_steps=config["max_steps"],
        per_device_train_batch_size=config["per_device_train_batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        num_generations=config["num_generations"],
        max_completion_length=config["max_completion_length"],
        temperature=config["temperature"],
        beta=config["beta"],
        bf16=config["bf16"],
        report_to="none",
    )
    trainer = GRPOTrainer(
        model=model,
        processing_class=processor,
        reward_funcs=product_audit_reward,
        args=training_args,
        train_dataset=dataset,
    )
    trainer.train()
    trainer.save_model(config["output_dir"])
    processor.save_pretrained(config["output_dir"])


if __name__ == "__main__":
    main()

