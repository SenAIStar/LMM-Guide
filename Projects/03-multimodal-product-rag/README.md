# 商品理解与问答（RAG + SFT + GRPO）

这部分代码对应商品理解与问答项目，保留检索、引用约束、训练数据和奖励计算几条主链路。

## 核心链路

`商品文本/图片 -> 权限与版本过滤 -> 稀疏/稠密召回 -> RRF 融合 -> 多模态精排 -> 引用约束生成 -> SFT -> GRPO`

## 代码目录

- `src/product_rag/catalog.py`：商品证据块、版本、时效和 ACL 过滤。
- `src/product_rag/retrieval.py`：稀疏与稠密召回、RRF 融合和精排接口。
- `src/product_rag/qwen_retrieval.py`：Qwen3-VL Embedding/Reranker 适配。
- `src/product_rag/qdrant_store.py`：向量检索与权限过滤条件。
- `src/product_rag/contracts.py`：回答 JSON、Claim 和 Citation 协议。
- `src/product_rag/service.py`：检索、生成、校验和转人工流程。
- `src/product_rag/training_data.py`：SFT 与 GRPO 样本组织。
- `src/product_rag/rewards.py`：格式、引用、字段一致性和拒答奖励。
- `src/product_rag/evaluation.py`：Recall、MRR、nDCG、引用和拒答指标。
- `scripts/train_sft.py`、`scripts/train_grpo.py`：LoRA SFT 与 GRPO 核心训练配置。

## 关键设计

检索前先做租户、ACL、商品范围和版本时效过滤。生成结果只接受单个 JSON 对象，每条 Claim 必须引用本次检索上下文中的证据块。GRPO 奖励沿用同一套协议校验，避免训练和线上验收各写一套口径。
