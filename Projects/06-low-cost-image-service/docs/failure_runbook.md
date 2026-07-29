# 故障演练与降级手册

## OOM

先记录失败 batch 的模型、shape、steps、adapter 集合和峰值显存。服务只在配置的预算内拆成更小 batch 重试；单请求仍 OOM 时直接失败，不做无限重试。`torch.cuda.empty_cache()` 只能释放未使用的缓存块，不能替代容量控制。

## 队列积压

查看兼容桶分布，而不是只看总队列长度。如果冷门 shape 或 adapter 组合长期凑不成 batch，达到 `max_delay_ms` 后应单独执行。超过租户配额或全局队列上限时在入口拒绝，避免请求已经超时却仍占 GPU。

## 编译抖动

若编译缓存持续增长，先检查 shape、dtype、adapter target/rank 是否变化。参考后端禁止 `compile_unet=true` 与动态 LoRA 同时开启。需要动态 adapter 时使用未编译 worker；需要编译收益时把固定模型、固定 adapter 和固定 shape 路由到专用 worker 池。

## 缓存污染

缓存仅保存质量与安全门禁通过的输出，目录按 tenant 隔离。缓存键覆盖模型和 adapter revision、文件 SHA-256、提示、负面提示、seed、shape、steps、scheduler、guidance、dtype、输出格式和策略版本。策略或权重升级后会自然形成新键，不覆盖旧结果。

## 依赖或模型加载失败

worker 未完成模型与 adapter 哈希校验时不进入 ready。滚动发布先启动新 revision 的 worker，预热并通过 smoke test 后再接流量；旧 revision 等待在途请求结束再下线。

