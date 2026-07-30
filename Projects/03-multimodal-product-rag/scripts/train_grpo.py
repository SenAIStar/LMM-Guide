from __future__ import annotations

import argparse
import json
from pathlib import Path

from product_rag.rewards import product_rag_reward


def load_config(path: str) -> dict:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    revision = config.get("model_revision", "")
    if revision == "REQUIRED_COMMIT_SHA" or len(revision) < 7:
        raise ValueError("model_revision must be an immutable Hugging Face commit SHA")
    if not Path(config["sft_adapter_path"]).is_dir():
        raise ValueError("sft_adapter_path must point to the completed SFT adapter")
    return config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/grpo.json")
    args = parser.parse_args()
    config = load_config(args.config)

    try:
        import torch
        from datasets import load_dataset
        from peft import LoraConfig, PeftModel
        from transformers import AutoModelForImageTextToText, AutoProcessor
        from trl import GRPOConfig, GRPOTrainer
    except ImportError as exc:
        raise RuntimeError("install requirements-ml.txt before GPU training") from exc

    processor = AutoProcessor.from_pretrained(
        config["model_id"], revision=config["model_revision"]
    )
    base_model = AutoModelForImageTextToText.from_pretrained(
        config["model_id"],
        revision=config["model_revision"],
        dtype=torch.bfloat16 if config["bf16"] else torch.float16,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(base_model, config["sft_adapter_path"]).merge_and_unload()
    dataset = load_dataset("json", data_files=config["dataset_path"], split="train")
    peft_config = LoraConfig(
        r=config["lora_r"],
        lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"],
        bias="none",
        target_modules=config["target_modules"],
        task_type="CAUSAL_LM",
    )
    training_args = GRPOConfig(
        output_dir=config["output_dir"],
        seed=config["seed"],
        learning_rate=config["learning_rate"],
        max_steps=config["max_steps"],
        per_device_train_batch_size=config["per_device_train_batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        gradient_checkpointing=config["gradient_checkpointing"],
        num_generations=config["num_generations"],
        max_completion_length=config["max_completion_length"],
        temperature=config["temperature"],
        beta=config["beta"],
        bf16=config["bf16"],
        remove_unused_columns=False,
        logging_steps=5,
        save_steps=50,
        report_to="none",
    )
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=product_rag_reward,
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
