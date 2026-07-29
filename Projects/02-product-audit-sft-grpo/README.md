# 多模态商品审核系统（SFT + RL）

这是一个面向算法岗面试的可运行项目：输入商品标题、类目、属性和多张商品图，输出结构化审核结论、风险代码、证据位置与命中的规则版本。项目不把“大模型能看图”当作完成标准，而是把数据、策略、训练、评测、服务兜底和复盘证据连成闭环。

所有业务效果和 GPU 性能指标默认标记为 `not_measured`。仓库内的样例分数仅用于验证代码链路，不能写成线上收益。

## 1. 项目目标

- 识别禁限售、联系方式导流、图文不一致和类目属性冲突等风险。
- 对每个风险返回图片序号、区域或整图证据、策略条款，而不是只给结论。
- 用 SFT 建立稳定格式和基础能力，再用 GRPO 优化规则遵循、风险召回与证据质量。
- 让确定性规则拥有最终否决权；模型解析失败、低置信度或冲突样本进入人工复核。

## 2. 审核协议

输出协议见 `docs/output_schema.json`。核心字段包括：

```json
{
  "schema_version": "audit-output.v1",
  "decision": "reject",
  "risk_codes": ["CONTACT_DIVERSION"],
  "evidence": [{
    "risk_code": "CONTACT_DIVERSION",
    "media_index": 0,
    "support": "region",
    "bbox": [0.12, 0.18, 0.83, 0.32],
    "policy_rule_id": "R-CONTACT-001"
  }],
  "policy_refs": ["R-CONTACT-001"],
  "explanation": "主图出现站外联系方式。"
}
```

`decision` 只有 `pass/reject/review`。坐标统一为 `[x1,y1,x2,y2]` 的 0 到 1 归一化值。禁止使用模型自己生成的 `confidence` 直接放行商品。

## 3. 数据设计

训练样本按 `product_id` 分组切分，避免同一商品的换图、裁剪图或改标题版本同时落入训练集与测试集。每条记录保留媒体 SHA-256、来源、授权信息、采集时间、策略快照和双人标注结果。

```powershell
python scripts/validate_data.py --input data/sample/audit_examples.jsonl --policy configs/policy.v1.json
python scripts/build_sft.py --input data/sample/audit_examples.jsonl --output artifacts/sft_train.jsonl
```

合成反事实样本可用于训练与开发集，例如叠加二维码、联系方式或制造图文冲突；冻结测试集只保留真实业务样本，并单独报告合成集结果。

## 4. 策略引擎

`src/product_audit/policy.py` 执行可审计的确定性规则。规则命中优先于模型判断，并将 `policy_version`、规则 ID 和输入摘要写入日志。策略切换使用 `effective_at` 和不可变版本，不在同一测试集中混用新旧口径。

```powershell
python scripts/run_baseline.py --input data/sample/audit_examples.jsonl --policy configs/policy.v1.json
```

## 5. SFT 冷启动

默认基座为 `Qwen/Qwen3-VL-4B-Instruct`。SFT 使用多图对话格式和 QLoRA；视觉语言模型要设置 `max_length=None`，或先证明截断不会删除图像 token。模型 revision 必须固定到提交哈希，避免训练和复现实验使用不同权重。

```powershell
python scripts/train_sft.py --config configs/sft.json --model-revision <commit_sha>
```

SFT 数据只训练审核协议内字段；解释文本限制长度，避免模型通过冗长描述掩盖错误结论。

## 6. GRPO 奖励

奖励顺序是“先合法、再合规、后优化”：

1. JSON 或协议无效时直接门控为负分。
2. 违反确定性硬规则时直接门控为负分。
3. 合法样本再计算决策代价、风险 F1、证据 IoU、策略引用和简短解释分。
4. 同组奖励方差为零时优势置零，不制造伪梯度。

```powershell
python scripts/score_rollouts.py --input data/sample/rollouts.jsonl --policy configs/policy.v1.json
python scripts/train_grpo.py --config configs/grpo.json --model-revision <commit_sha>
```

`scripts/train_grpo.py` 对接 TRL 的 VLM GRPO 输入协议。VeRL 只作为大规模训练的升级路径，必须先通过当前 Qwen3-VL 版本的 rollout、图像预处理和 checkpoint 兼容性测试。

## 7. 训练配置

`configs/sft.json` 与 `configs/grpo.json` 是显存友好的起点，不是最优参数。正式实验至少记录：代码提交、模型 revision、数据版本、策略版本、随机种子、GPU 型号、显存峰值和 wall-clock 时间。

依赖分为 CPU 合同测试和 GPU 训练两层：

```powershell
python -m pip install -e .
python -m unittest discover -s tests -v
python -m pip install -r requirements-ml.txt
```

## 8. 评测与消融

```powershell
python scripts/evaluate_predictions.py `
  --gold data/sample/audit_examples.jsonl `
  --predictions data/sample/predictions.jsonl `
  --policy configs/policy.v1.json
```

主指标包括协议合法率、分风险 Macro-F1、高风险漏审率、自动拒绝误杀率、证据 Precision/Recall、规则冲突率、人工复核率和外部校准置信度的 ECE。所有指标都按类目、风险、图片数量、OCR 密度和新旧策略切片。

最少完成四组消融：仅规则、仅 SFT、SFT+GRPO、去除证据奖励。详细口径见 `docs/evaluation.md`。

## 9. 部署与兜底

在线链路先跑硬规则，再调用多模态模型，最后执行协议校验和决策路由。硬规则命中可直接拒绝；模型输出无效、与规则冲突或外部置信度低时进入人工复核。只有校准集上达到阈值的 `pass/reject` 才允许自动执行。

服务代码见 `src/product_audit/service.py`，Qwen3-VL 推理适配见 `src/product_audit/qwen3vl_adapter.py`。线上必须监控协议失败率、风险分布漂移、人工复核率、规则冲突率和各风险的延迟分位数。

## 10. 项目交付

建议简历表述：

> 设计多图商品审核系统，基于 Qwen3-VL 完成 SFT 与组相对策略优化；实现策略硬门控、结构化证据协议、按商品分组的数据防泄漏、分风险评测与人工复核兜底。离线效果以可复现实验表为准，线上指标未测量。

面试时应能现场解释：为什么不能只报 Accuracy、为什么模型自报置信度不能用于放行、GRPO 零方差组如何处理、策略更新后怎样避免标签口径污染，以及误杀和漏审的代价为何不同。

参考资料：

- [Qwen3-VL 官方仓库](https://github.com/QwenLM/Qwen3-VL)
- [TRL SFTTrainer](https://huggingface.co/docs/trl/sft_trainer)
- [TRL GRPOTrainer](https://huggingface.co/docs/trl/grpo_trainer)
- [VeRL 多模态示例](https://verl.readthedocs.io/en/latest/examples/multi_modal_example.html)
- [DeepSeekMath / GRPO](https://arxiv.org/abs/2402.03300)

