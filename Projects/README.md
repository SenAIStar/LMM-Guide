# LMM-Guide 多模态项目集

这组工程把讲义里的 7 个多模态项目收敛为同一套可审计方法：业务目标、数据合同、训练/系统实现、离线评测、消融、安全边界、简历表达和面试追问。

## 当前可运行范围

- `lmm_core/`：不依赖第三方库的数据校验、检索指标、奖励函数和安全门禁。
- `tools/demo.py`：7 个项目的轻量演示，不加载模型权重。
- `tools/validate_configs.py`：校验全部项目配置和验收指标。
- `tests/`：覆盖输入合同、指标、奖励和飞行安全策略。
- 各项目的模型训练、向量库、图像生成和 PX4 SITL 需要按项目 README 安装 GPU 依赖、准备数据与权重后再执行。

文档中的数值是“参考验收目标”，不是本仓库已经跑出的实验结果。只有生成带硬件、数据版本、随机种子和提交哈希的评测报告后，才能改写成“实测结果”。

## 项目目录

| 目录 | 项目 | 主要技术 |
|---|---|---|
| `01-vlm-understanding-sft` | 多模态理解 SFT | Qwen3-VL + LLaMA-Factory + LoRA |
| `02-product-audit-sft-grpo` | 多模态商品审核 | Qwen3-VL + SFT + VeRL GRPO |
| `03-multimodal-product-rag` | 商品理解与问答 | Qwen3-VL-Embedding/Reranker + RAG + SFT + GRPO |
| `04-drone-agent-sitl` | 多模态无人机 Agent | Qwen3-VL + PX4 SITL + MAVSDK + Safety Gate |
| `05-controlnet-lora-generation` | 可控图像生成 | SDXL + ControlNet + LoRA + Diffusers |
| `06-low-cost-image-service` | 低成本图像生成服务 | Diffusers + 批处理 + 缓存 + 显存治理 |
| `07-janus-audit-generation` | 审核与生成辅助 | Janus-Pro + 结构化 SFT + 文本决策 GRPO |

## 快速验证

```powershell
python -m unittest discover -s tests -v
python tools/validate_configs.py
python tools/demo.py --project all
```

## 结果证据要求

每次正式实验至少保存：`data_version`、`model_revision`、`config_hash`、`git_commit`、`seed`、GPU 型号、样本级预测、聚合指标和失败样本。训练日志与评测集必须隔离，人工验收集不得回流到训练。

## 官方实现依据

- Qwen3-VL: https://github.com/QwenLM/Qwen3-VL
- Qwen3-VL-Embedding/Reranker: https://github.com/QwenLM/Qwen3-VL-Embedding
- LLaMA-Factory: https://github.com/hiyouga/LlamaFactory
- VeRL GRPO: https://github.com/volcengine/verl/tree/main/examples/grpo_trainer
- Diffusers ControlNet/LoRA: https://huggingface.co/docs/diffusers
- Janus: https://github.com/deepseek-ai/Janus
- MAVSDK Python: https://mavsdk.mavlink.io/main/en/python/quickstart.html
- Amazon Berkeley Objects: https://www.amazon.science/code-and-datasets/amazon-berkeley-objects-abo-dataset

