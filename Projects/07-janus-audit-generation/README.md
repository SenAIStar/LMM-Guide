# 5.1 内容审核与生成辅助系统（Janus + SFT + GRPO）

这个工程把同一个多模态入口拆成两条受控链路：Janus-Pro 理解侧读取已授权素材并输出带证据的审核 JSON；只有通过确定性策略引擎的 `pass/revise` 结果，才会转换成去风险的生成 brief，再交给 Janus 图像生成侧。生成图片必须再次走审核，模型不能自行绕过策略。

## 能力边界

Janus 官方仓库提供 Janus-Pro 推理和演示代码，没有可直接复用的完整 SFT/GRPO 训练流水线。本仓库因此交付三层内容：可执行的 CPU 侧数据、合同、策略、奖励和评测代码；需要 GPU 与官方依赖的 Janus 冻结基线适配器；明确标为实验性的结构化 SFT 与文本决策 GRPO 配置。GRPO 只优化审核 JSON 和安全 brief，不宣称对图像 token 做端到端强化学习。

`project.json` 中的验收目标均为 `not_measured`。样例图是自建 PPM 色块，只用于验证哈希、合同和脚本，不代表真实业务效果。

## 工程结构

```text
configs/                 政策、SFT、GRPO 和评测配置
data/sample/             可追溯的合成样例、预测与 rollout
src/janus_audit/         合同、策略、奖励、GRPO、评测与 Janus 适配器
scripts/                 数据转换、rollout 打分、离线评测和 GPU 基线入口
tests/                   CPU 可运行的核心测试
docs/                    评测协议与失败分析
```

## CPU 侧复现

Windows PowerShell：

```powershell
python -m unittest discover -s tests -v
python scripts/validate_data.py
python scripts/build_sft.py
python scripts/score_rollouts.py
python scripts/evaluate_predictions.py
```

预期结果是测试通过、样例数据校验通过，并在 `artifacts/` 产生 SFT 与 GRPO 中间文件。样例预测与标注相同，所以示例评测是管线自检，不得作为模型结果对外引用。

## GPU 基线

先在独立 CUDA 环境安装 `requirements-ml.txt`，审核 Janus 官方代码和模型许可，然后把配置中的占位 revision 换成实际审过的 Hugging Face commit。不要直接使用浮动分支部署。

```powershell
python scripts/run_janus_baseline.py `
  --revision <reviewed_commit> `
  --image data/sample/assets/demo_ad.ppm `
  --instruction "按 AuditResult 合同审核素材，只输出 JSON"
```

生成侧只接收 `build_generation_request()` 返回的请求。示例：

```powershell
python scripts/run_janus_baseline.py `
  --revision <reviewed_commit> `
  --generate-prompt "抽象配色海报，不包含人物、真实品牌或受监管承诺" `
  --output artifacts/janus_generation.png
```

## 数据与训练顺序

1. 素材入库时写入 SHA-256、`license_id`、`group_id`、`policy_version`，近重复素材按 group 切分，禁止跨 train/test 泄漏。
2. 先跑冻结 Janus-Pro 零样本基线，保存原始输出、JSON 合法率和失败桶。
3. 结构化 SFT 只学习严格合同、证据和拒答；模块名必须按固定 revision 重新核对。
4. GRPO 每个 prompt 采样多个文本候选，奖励拆成决策、风险标签、证据、brief 和复核校准；硬策略违规直接记 `-1`。
5. 同时评估审核、理解 holdout、生成 holdout。任一跨任务回归超阈值都不发布。

## 面试时应能解释

- Janus 为什么“统一”但仍需要把理解审核和图像生成分别评测。
- 为什么文本决策 GRPO 不能被描述为图像质量 RL。
- 为什么 Macro-F1 之外还要看高风险召回、证据 precision、coverage/selective error 和策略违规率。
- 零方差 rollout 组为什么没有相对优势信号，如何从采样温度、候选数和奖励分辨率排查。
- 为什么模型给出 `pass` 仍不能直接发布，硬策略、人工复核和生成后复审分别挡什么风险。

详细口径见 [评测协议](docs/evaluation.md) 与 [失败分析](docs/failure_analysis.md)。
