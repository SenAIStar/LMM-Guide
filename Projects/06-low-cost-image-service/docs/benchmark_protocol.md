# GPU 压测与消融协议

这份协议只定义怎么测，不附带虚构结果。`FakeBackend` 只验证控制面逻辑，不能写入 GPU 性能表。

## 1. 固定项

- 固定 GPU 型号、驱动、CUDA、PyTorch、Diffusers 和容器镜像摘要。
- 固定基础模型 commit、VAE、LoRA 文件 SHA-256、调度器、提示集和质量门禁版本。
- 固定请求分布，不把所有请求都简化为同一个分辨率和步数。
- 每个方案先预热，编译耗时、模型加载耗时另行记录，不混入稳态延迟。
- 每个方案至少重复三轮；原始逐请求日志和聚合脚本一起留存。

## 2. 基线和实验组

| 组别 | batch | cache | compile | VAE | offload | 目的 |
|---|---:|---|---|---|---|---|
| B0 | 1 | off | off | default | off | 单请求基线 |
| B1 | dynamic | off | off | default | off | 只测 micro-batching |
| B2 | dynamic | exact | off | default | off | 测完全一致请求缓存 |
| B3 | dynamic | off | on | default | off | 固定 shape 编译收益 |
| B4 | dynamic | off | off | slicing | off | 多图 VAE 解码显存 |
| B5 | dynamic | off | off | tiling | off | 大分辨率显存与接缝风险 |
| B6 | dynamic | off | off | default | model | 显存换传输开销 |

不要一次把 compile、量化、offload 和 attention backend 全开后只报总收益。组合方案可以作为最终候选，但单项消融必须保留。

## 3. 请求日志最小字段

每条请求记录 `request_id`、租户、输入指纹、模型与 adapter revision、入队时间、出队时间、开始/结束时间、batch ID、batch size、缓存命中、重试次数、OOM、质量门禁结果和输出哈希。batch 尝试还要单独记录 GPU 时间，失败尝试不能从成本中删除。

## 4. 指标

- 有效吞吐：`质量门禁通过图片数 / 聚合 GPU 小时`。
- 端到端延迟：从接收请求到返回最终状态，报告 P50/P95/P99。
- 排队延迟：从接收请求到 batch 开始执行，报告 P95。
- OOM 尝试率：`OOM batch 尝试数 / 全部 batch 尝试数`。
- 质量门禁通过率：`通过数 / 已生成并接受检查的图片数`。
- 单张有效图片 GPU 成本：`全部尝试 GPU 小时 * 实际 GPU 小时价格 / 通过数`。
- 峰值显存：每个 worker 在稳态窗口内的 `max_memory_allocated` 和 `max_memory_reserved`，两者分开报告。

## 5. 上线门槛

仓库里的 `1.6x`、`0.1%` 和 `95%` 都是参考验收目标，状态为 `not_measured`。只有真实 GPU 报告、原始日志、质量样本和环境指纹齐全时，才能把结果状态改为 `measured`。

