# 可控图像生成系统（ControlNet + LoRA）

这个项目把“结构可控”和“业务风格适配”拆成两条独立能力：SDXL 负责通用生成，预训练 Canny ControlNet 约束商品轮廓与版式，LoRA 学习已授权的商品域或品牌视觉。主线先复用预训练 ControlNet，只训练 LoRA；只有预训练控制器在业务验证集上持续失败，才进入可选的 ControlNet 微调。

当前仓库已经实现数据合同、条件图构建、训练参数生成、联合推理、结构评测和运行指纹，但没有在本机下载权重或执行 GPU 训练。因此 `project.json` 中的数值是参考验收门槛，不是实验结果。

## 1. 项目边界

具体任务：输入一张商品 Canny 边缘图和一条营销描述，生成轮廓与布局可复现、视觉风格符合授权素材域的 1024×1024 商品图。

ControlNet 不负责记住品牌风格，LoRA 也不负责精确复制边缘。两者的验收必须分开：

| 能力 | 主要输入 | 主要输出 | 核心验收 |
|---|---|---|---|
| SDXL 基线 | prompt + seed | 通用商品图 | 文本一致、基本质量 |
| ControlNet | condition image + scale | 空间约束残差 | Edge F1、轮廓错位率 |
| LoRA | 授权图像 + caption | 小型适配权重 | 风格/主体相似、过拟合检查 |
| 联合推理 | prompt + condition + 两种 scale | 最终素材 | 结构、风格、可用性与安全 |

以下内容不在当前结果声明中：线上转化提升、真实吞吐、实际显存、商业投放效果和任何已达到的百分比指标。它们需要在指定硬件和冻结数据版本上重新测量。

## 2. 为什么采用两阶段方案

ControlNet 冻结基础扩散模型，并用可训练分支接收边缘、深度、姿态或分割等空间条件。它解决“物体放在哪里、轮廓如何保持”。LoRA 冻结原权重，只训练低秩更新，解决“生成结果更像哪一类商品或视觉风格”。

工程顺序如下：

1. 运行 SDXL 基线，建立不带控制与不带适配的结果。
2. 加载预训练 Canny ControlNet，扫描 `controlnet_conditioning_scale`，先证明结构约束有效。
3. 只在已授权训练图上训练 LoRA，检查不同 rank、步数和 scale 的风格收益与过拟合。
4. 联合加载 ControlNet 与 LoRA，做二维 scale 网格，不从几张最好样例挑参数。
5. 只有预训练 ControlNet 在特定商品边缘域上无法过门槛，才用成对数据继续微调 ControlNet。

这样可以把问题定位到控制器、适配器、数据或采样参数，而不是把两个训练过程混在一起后无法解释失败原因。

## 3. 目录结构

```text
05-controlnet-lora-generation/
├─ configs/                  # LoRA、可选 ControlNet、推理与消融配置
├─ data/                     # 生产数据不入库；sample 仅验证代码
├─ docs/evaluation.md        # 指标、盲测与结果纪律
├─ scripts/                  # 数据、训练命令、推理和评测入口
├─ src/controlnet_lora/      # 数据合同、条件图、指标与运行指纹
├─ tests/                    # 不下载模型也能运行的单元测试
├─ project.json              # 项目状态与参考门槛
└─ requirements-ml.txt       # GPU 路径依赖
```

## 4. 数据合同与防泄漏

最终 `manifest.jsonl` 每行使用同一协议：

```json
{"sample_id":"sku_001","image":"images/sku_001.png","conditioning_image":"conditions/sku_001.png","text":"studio product photo of an authorized red package on a clean background","subject_id":"sku_red","capture_group":"shoot_202607_batch01","license_id":"LIC-2026-001","split":"train","condition_type":"canny","sha256_image":"<64-hex>","sha256_conditioning":"<64-hex>"}
```

硬约束：

- 目标图、条件图和 caption 一一对应，尺寸一致。
- 同一商品、同一视频抽帧或同一拍摄批次使用同一个 `capture_group`，只能落在一个 split。
- 每张图必须能追溯到授权台账的 `license_id`；网页可见不代表允许训练。
- 文件哈希写入清单，训练开始后不允许静默替换图片。
- Validation 用于选参数，Test 只在方案冻结后运行；人工盲测集不回流训练。

验证示例数据：

```powershell
python scripts/validate_manifest.py `
  --manifest data/sample/manifest.jsonl `
  --root data/sample
```

生成生产 Canny 条件图：

```powershell
python scripts/build_canny.py `
  --input-dir data/raw/images `
  --output-dir data/processed/conditions `
  --low 100 `
  --high 200
```

Canny 阈值属于数据版本。改变阈值后必须重算条件图哈希，并作为消融变量重新评测。

## 5. LoRA 训练主线

先从最终 manifest 提取训练集，生成官方 Diffusers `ImageFolder` 所需的 `metadata.jsonl`：

```powershell
python scripts/prepare_lora_imagefolder.py `
  --manifest data/manifest.jsonl `
  --root data `
  --output data/lora_imagefolder
```

训练前克隆 Diffusers，并把 `configs/lora_train.json` 中的 `diffusers_commit` 替换成实际验证过的提交。项目脚本只生成命令，不会替用户直接启动长时训练：

```powershell
python scripts/print_train_command.py `
  --stage lora `
  --config configs/lora_train.json `
  --diffusers-root D:\src\diffusers
```

初始配置使用 UNet attention LoRA、rank 16、BF16、gradient checkpointing 和 3000 steps。这些是起跑参数，不是最佳参数。BF16 需要兼容硬件；不支持时改 FP16，并重新记录硬件和配置哈希。是否训练 text encoder 必须单独做消融，不能默认开启。

需要保留：最终 `safetensors`、每个 checkpoint、训练日志、验证图、数据版本、Diffusers commit、基础模型 revision、随机种子和配置文件。只展示最终图片而没有这些证据，无法判断收益来自训练还是挑图。

## 6. 预训练 ControlNet 与可选微调

主线加载 `diffusers/controlnet-canny-sdxl-1.0`。先在冻结验证集上扫描 scale，并按商品类型、背景复杂度和边缘密度分桶。如果低边缘密度时轮廓丢失、高密度时纹理被过度复制，先调整条件图和 scale，不要立刻训练新 ControlNet。

只有以下情况才进入微调：

- 业务条件图与预训练 Canny 分布明显不同，且调整阈值与 scale 后仍失败。
- 需要新的条件类型，现有权重不支持。
- 失败在多个种子、多个分桶上稳定出现，不是个别坏样例。

可选训练数据要构造成带类型的 `DatasetDict`。本地脚本可以保存离线 Arrow 数据，也可以在用户明确登录后推送到私有 Hub 数据集：

```powershell
python scripts/build_hf_controlnet_dataset.py `
  --manifest data/manifest.jsonl `
  --root data `
  --output data/hf_dataset `
  --hub-id ORG/PRIVATE_CONTROLNET_DATASET
```

然后生成官方 `train_controlnet_sdxl.py` 命令：

```powershell
python scripts/print_train_command.py `
  --stage controlnet `
  --config configs/controlnet_train.json `
  --diffusers-root D:\src\diffusers
```

如果不需要自训 ControlNet，不运行这一段，也不要在简历里写“训练了 ControlNet”。

## 7. 联合推理与复现

`configs/inference.json` 固定基础模型、ControlNet、LoRA、两种 scale、prompt、negative prompt、步数、分辨率和 seed。正式运行前把三个 revision 占位符替换为真实模型 revision 或文件哈希。

```powershell
python scripts/generate.py `
  --config configs/inference.json `
  --control-image data/eval/conditions/sku_101.png `
  --lora artifacts/lora `
  --output outputs/sku_101_seed_20260729.png
```

推理代码使用 `StableDiffusionXLControlNetPipeline`，通过 `load_lora_weights()` 加载 LoRA，并用 `set_adapters()` 设置 LoRA scale。生成图旁边会写出同名 `.run.json`，记录配置指纹、seed 和模型 revision。`result_status` 初始为 `generated_not_evaluated`，评测完成前不能写成已验收。

scale 的解释：

- ControlNet scale 过低：结构漂移，商品轮廓和版式不稳定。
- ControlNet scale 过高：边缘被僵硬复制，纹理细节、光照和背景自由度下降。
- LoRA scale 过低：风格或主体特征不足。
- LoRA scale 过高：训练图记忆、颜色偏置、触发词污染或结构让位于风格。

## 8. 评测与消融

完整协议见 `docs/evaluation.md`。最低限度必须同时报告：

| 维度 | 指标/方法 | 不能替代什么 |
|---|---|---|
| Canny 结构 | 带像素容差的 Edge Precision/Recall/F1 | 不能代表美观或文本一致 |
| 文本一致 | CLIP 类相似度 | 不能证明轮廓、身份或安全 |
| 分布质量 | FID/KID/CMMD + 相同样本量 | 小样本下不能单独决策 |
| 风格/主体 | 冻结视觉编码器相似度 + 分桶 | 不能替代人工可用性判断 |
| 人工质量 | 双盲偏好、缺陷标签、置信区间 | 不能只看几张最佳样例 |
| 工程 | P50/P95、峰值显存、失败率、成本 | 必须注明硬件与并发 |
| 安全 | 错误放行率、误拒率 | 两者必须分开 |

必做矩阵：Base、ControlNet only、LoRA only、ControlNet + LoRA；联合方案扫描 `control_scale ∈ {0.5, 0.8, 1.0}` 与 `lora_scale ∈ {0.5, 0.7, 0.9}`。固定同一组 prompt、condition 和 seeds，再增加 rank、训练步数、Canny 阈值、分辨率与采样步数消融。

结构评测命令：

```powershell
python scripts/evaluate_edges.py `
  --prediction-edge outputs/sku_101.edge.png `
  --reference-edge data/eval/conditions/sku_101.png `
  --tolerance 1.5
```

对深度、姿态和分割控制，分别换成 AbsRel/RMSE、PCK 和 mIoU。不要把所有控制类型混成一个没有定义的“结构相似度”。

## 9. 失败分析

每个失败样本至少标注一类原因：

- 条件图问题：断边、背景纹理过密、尺寸/裁剪不一致、阈值变化。
- ControlNet 问题：scale 过低导致漏控，过高导致僵硬或复制噪声。
- LoRA 问题：欠拟合、过拟合、触发词污染、颜色/视角偏置。
- 联合冲突：风格把轮廓推偏，或结构约束压制材质与光照。
- 文本问题：描述与条件图冲突，商品数量或朝向不一致。
- 基础模型限制：文字渲染、细小 Logo、手部、人脸、多对象关系失败。
- 合规问题：未授权商标、人物肖像、受限提示或训练样本删除请求。

失败修复要从可定位变量开始：先重放相同 seed 和配置，检查条件图；再单独关掉 LoRA 或 ControlNet；最后才考虑改训练数据或模型。无法复现的失败不应进入“已修复”统计。

## 10. 测试与结果纪律

不安装模型依赖也能运行：

```powershell
python -m unittest discover -s tests -v
python scripts/validate_manifest.py --manifest data/sample/manifest.jsonl --root data/sample
```

这些测试只证明数据校验、泄漏检测、Edge F1 和配置指纹逻辑可运行，不证明模型质量。正式结果还必须附带：

- 数据清单哈希与授权台账版本。
- Base、Control-only、LoRA-only、Joint 的样本级预测。
- GPU、CUDA、PyTorch、Diffusers commit 和模型 revision。
- 每个 seed 的指标，不只给平均值。
- 人工盲测任务、评审人数、随机化方式和置信区间。
- 失败样本库与安全集的错误放行/误拒明细。

参考门槛 `Edge F1 ≥ 0.72`、`盲测偏好 ≥ 65%`、`错误放行率 ≤ 5%` 只是项目验收模板。没有报告和原始证据时，简历中只能写“设计了该验收口径”，不能写“达到”。

## 11. 面试说明口径

可诚实表述为：

> 基于 SDXL 构建商品图可控生成链路，使用预训练 Canny ControlNet 约束轮廓、LoRA 适配授权商品域；定义成对数据合同和拍摄批次级防泄漏，完成 Base/Control-only/LoRA-only/Joint 消融、二维 scale 扫描、结构评测与运行指纹。当前仓库提供可复现实验骨架，实际 GPU 指标以评测报告为准。

高频追问：

1. 为什么不直接全量微调 SDXL？要比较训练成本、权重管理、灾难性遗忘和多品牌切换。
2. ControlNet 与 LoRA 分别改变什么？前者注入空间条件残差，后者学习低秩参数更新。
3. 两种 scale 冲突怎么办？固定 prompt/seed 做二维网格，同时看结构、风格和人工偏好。
4. 为什么不能只看 CLIPScore？它偏语义对齐，不能证明边缘忠实、品牌一致或视觉可用。
5. 如何防数据泄漏？按主体和拍摄批次分组，近重复检测后切分，Test 只在方案冻结后运行。
6. 什么时候值得自训 ControlNet？只有预训练控制器跨分桶稳定失败，且数据/参数问题已排除。
7. 如何处理删除请求？通过 `license_id` 和样本哈希定位数据，重建受影响 LoRA 版本并下线旧权重。

## 12. 官方实现依据

- ControlNet paper: https://arxiv.org/abs/2302.05543
- LoRA paper: https://arxiv.org/abs/2106.09685
- Diffusers ControlNet training: https://huggingface.co/docs/diffusers/training/controlnet
- Diffusers SDXL ControlNet script: https://github.com/huggingface/diffusers/blob/main/examples/controlnet/train_controlnet_sdxl.py
- Diffusers SDXL LoRA script: https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image_lora_sdxl.py
- Diffusers LoRA loading API: https://huggingface.co/docs/diffusers/api/loaders/lora
- SDXL base model card and license: https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0

Diffusers 的 `main` 分支脚本会变化。正式训练必须把仓库 commit 写进配置和报告，并在该 commit 上重新生成训练命令。
