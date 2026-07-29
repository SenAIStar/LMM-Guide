#!/usr/bin/env bash
set -euo pipefail

: "${MODEL_REVISION:?Set MODEL_REVISION to a reviewed immutable commit}"
: "${ADAPTER_PATH:?Set ADAPTER_PATH to the released LoRA directory}"

vllm serve Qwen/Qwen3-VL-4B-Instruct \
  --revision "${MODEL_REVISION}" \
  --enable-lora \
  --lora-modules "product=${ADAPTER_PATH}" \
  --limit-mm-per-prompt '{"image": 4, "video": 0}'
