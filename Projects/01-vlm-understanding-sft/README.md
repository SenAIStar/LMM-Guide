# 2.1 多模态理解（SFT）

我长期参与算法岗位面试。真正能拉开候选人差距的，通常不是简历里多写一个模型名，而是能否把数据、训练、评测、部署和失败分析讲清楚，并拿出可复现代码与实验记录。这个项目按真实面试追问设计：输入一张或多张同一商品的图片，输出固定 JSON，包含商品类型、可见颜色、可见材质、图中文字、证据所在图片和处理决定。第一版只处理静态商品图，不做视频、直播流、价格预测、销量预测或内容审核。

当前仓库已经通过本地数据、合同、溯源、评测和路由测试，但没有下载 ABO 全量数据，没有执行 GPU Zero-Shot、LoRA 训练、官方外部基准或压力测试。因此所有效果、延迟和显存指标都是 `not_measured`。样例预测与样例标签一致，只能证明评测代码能运行，不能当成模型结果。

## 1. 项目定位

很多多模态项目停在“上传一张图，模型返回一段话”的 Demo 层，面试时很快就会遇到四个问题：输出怎样验收、训练数据是否泄漏、SFT 是否真的有效、线上错误如何兜底。本项目把任务收敛为商品目录图理解，并把每个简历技术点落实成可检查的交付物：

| 能力 | 项目设计 | 面试证据 |
|---|---|---|
| 任务定义 | 静态商品图转固定 JSON | Schema、字段边界、失败路由 |
| 数据工程 | ABO 候选标签、人工复核、商品级切分 | 许可快照、媒体哈希、泄漏检查 |
| 模型训练 | Qwen3-VL Zero-Shot 与 LoRA SFT | 固定底座、受控变量、训练配置 |
| 离线评测 | 字段、证据、幻觉、切片和消融 | 冻结测试集、逐样本预测、评测报告 |
| 在线服务 | vLLM 推理与业务校验分层 | 输入限额、人工复核、审计日志 |
| 项目答辩 | 结论回到代码和实验制品 | 仓库、日志、预测、报告、失败样本 |

这个项目不能保证面试结果，但能让简历中的技术点经得起追问。当前代码统一使用 `Qwen/Qwen3-VL-4B-Instruct`、`AutoModelForImageTextToText` 与 `AutoProcessor`，底座 revision 必须显式固定。

## 2. 输出协议

输出遵循 `vlm_product.schema.v1`：

```json
{
  "schema_version": "1.0",
  "product_type": "chair",
  "attributes": {
    "color": ["black"],
    "material": ["wood"]
  },
  "visible_text": [],
  "evidence": [
    {"field": "product_type", "media_index": 0, "support": "image_level"},
    {"field": "attributes.color", "media_index": 0, "support": "image_level"},
    {"field": "attributes.material", "media_index": 0, "support": "image_level"}
  ],
  "decision": "accept"
}
```

`product_type` 看不清时写 `unknown`；颜色或材质看不清时使用空数组。没有区域标注时只允许 `image_level` 证据，不生成虚假的框坐标。输出出现未知关键字段时必须为 `review`；JSON、字段、证据索引或枚举不合法时由服务层 `reject`。

严格合同解决三个问题：训练样本可以在入库前校验；Zero-Shot 与 LoRA 能在同一评测器下比较；线上系统可以把格式错误、视觉不确定与成功结果分开路由。

## 3. 数据设计

主原型使用 Amazon Berkeley Objects（ABO）。官方数据页列出 147,702 个商品与 398,212 张目录图。该页面当前写 CC BY 4.0，而 CVPR 2022 论文和 AWS Open Data Registry 写 CC BY-NC 4.0。仓库不替使用者消解这个冲突：正式下载后必须保存压缩包中的许可证文件、下载时间、来源 URL、压缩包 SHA-256、许可证 SHA-256 与署名文本；能否用于具体业务要按实际快照和法律要求判断。

ABO 元数据只是候选标签。商品标题写了“真皮”不代表照片能够证明材质；容量、重量和功能也常常不可见。`abo_candidate()` 默认产生 `review_required=True` 的候选记录，只有人工确认“图像支持该字段”后，才能转换成训练样本。

训练样本还要满足：图片能解码；图片 SHA-256 与记录一致；同一 `group_id` 只进入一个 split；相同文件哈希不能跨 split；感知近重复在正式数据构建中另行聚类审核。仓库没有用简化哈希冒充完整近重复算法。

## 4. 数据格式

当前 LLaMA-Factory 的多模态 ShareGPT 示例使用 `messages`、`role`、`content` 和 `images`。每个 `<image>` 必须与 `images` 中的一项一一对应：

```json
{
  "sample_id": "demo_train_001",
  "group_id": "product_demo_001",
  "split": "train",
  "images": ["sample/assets/demo_product.ppm"],
  "media_sha256": ["bfb01feb120b746a4e277cf1f4dadd71b3bdb58c05e9e2df1877910cd6ae6a9b"],
  "messages": [
    {"role": "user", "content": "<image>\n只根据图片抽取字段，无法确认的字段留空。"},
    {"role": "assistant", "content": "{\"schema_version\":\"1.0\",...}"}
  ],
  "source": {
    "dataset": "synthetic_format_demo",
    "snapshot_id": "demo-v1",
    "license_id": "CC0-1.0",
    "source_uri": "local-generated"
  },
  "review_required": false
}
```

`data/dataset_info.json` 是训练时实际读取的注册表；`configs/dataset_info.json` 是便于同步到外部 LLaMA-Factory checkout 的同内容副本。校验器会拒绝非 `messages/role/content` 格式、图片占位符数量错误、assistant 答案中的 `<image>`、不完整溯源和未经审核的训练记录。

## 5. 基线推理

先跑底座，再加载 LoRA。两组实验固定模型 revision、处理器、prompt、像素上限、冻结测试集、解码参数和评测代码，只改变 adapter。`scripts/run_zero_shot.py` 使用 Qwen3-VL 官方 Transformers 路径：

```powershell
python scripts/run_zero_shot.py data/sample/assets/demo_product.ppm `
  --revision <reviewed-model-commit>
```

运行前安装推理依赖：

```powershell
pip install -e ".[inference]"
```

官方 README 要求 Qwen3-VL 使用 `transformers>=4.57.0`，并给出 `qwen-vl-utils==0.0.14`。当前适配器采用处理器的 `apply_chat_template(..., tokenize=True, return_dict=True)`，生成后裁掉输入 token 再解码。它是惰性加载，单元测试不需要下载模型。

## 6. LoRA SFT

`configs/qwen3vl_lora_sft.yaml` 以 LLaMA-Factory 官方 Qwen3-VL LoRA 示例为起点，保留 `Qwen/Qwen3-VL-4B-Instruct`、`qwen3_vl_nothink`、LoRA rank 8、`lora_target: all`、学习率 `1e-4`、batch 1 与梯度累积 8。这里的 `1e-4` 是官方示例起点，不是已经验证的最优值。

训练前必须把以下占位符改成不可变值：

- `model_revision`：审阅过的 Hugging Face 模型 commit。
- `llamafactory_commit`：实际安装的 LLaMA-Factory commit。
- `dataset_snapshot` 与 `dataset_license_sha256`：本次训练使用的数据和许可证快照。

```powershell
git clone https://github.com/hiyouga/LlamaFactory.git
git -C LlamaFactory checkout <reviewed-llamafactory-commit>
pip install -e "./LlamaFactory[torch,metrics]"

llamafactory-cli train configs/qwen3vl_lora_sft.yaml
```

训练日志至少记录数据哈希、split salt、底座 revision、依赖版本、图像像素上限、LoRA 参数、随机种子、有效 batch、loss、学习率、梯度范数、吞吐、峰值显存、checkpoint 与选择理由。不能只保留最好的一次运行，也不能用训练 loss 代替冻结测试集结果。

## 7. 评测与消融

主评测按层拆开：

| 层 | 指标 | 回答的问题 |
|---|---|---|
| 结构 | JSON/schema 有效率、缺失预测数 | 输出能否被系统消费 |
| 字段 | 商品类型和属性的 Exact Match、Macro-F1 | 关键字段是否正确 |
| 可置信度 | unsupported attribute rate、证据覆盖率、复核率 | 是否编造不可见属性 |
| 外部诊断 | COCO Caption、TextVQA、A-OKVQA、POPE | 通用描述、OCR、知识问答和对象幻觉是否退化 |
| 系统 | TTFT、P50/P95、吞吐、峰值显存、错误率 | 固定硬件与并发下能否服务 |

COCO、TextVQA、A-OKVQA 和 POPE 使用各自官方脚本与协议；本仓库的字段评测器不能冒充官方分数。建议的消融包括 Zero-Shot/LoRA、元数据弱标签/人工审核标签、单图/多图、低/高像素上限、冻结/可训练视觉模块，以及自由文本/固定 JSON。每次只改一个主要变量。

## 8. 部署与兜底

Qwen3-VL 官方 README 推荐 `vLLM>=0.11.0`。LoRA 服务命令如下，实际 adapter 路径和模型 revision 要替换为发布制品：

```bash
vllm serve Qwen/Qwen3-VL-4B-Instruct \
  --revision <reviewed-model-commit> \
  --enable-lora \
  --lora-modules product=./saves/qwen3-vl-4b/lora/product-sft \
  --limit-mm-per-prompt '{"image": 4, "video": 0}'
```

vLLM 文档说明 `--limit-mm-per-prompt` 默认对每种模态可到 999，因此这里显式收紧图片和视频数量。若服务允许远程媒体 URL，应设置 `--allowed-media-domains`，并关闭可绕过域名限制的重定向，降低 SSRF 风险。多模态模型的视觉塔与连接器 LoRA 仍属于实验支持，发布前要检查当前模型的组件支持状态；不能因为训练配置写了 `lora_target: all` 就假定服务端已加载所有视觉侧权重。

应用链路为：请求校验 -> 图片解码与哈希 -> 模型推理 -> JSON 提取 -> schema 与证据校验 -> 接受、人工复核或拒绝。语义不确定不靠反复采样“重试到成功”；只对瞬时系统错误做有限重试。

## 9. 项目答辩

错误桶至少包括商品类型混淆、光照导致颜色偏差、材质不可见、包装 OCR 漏读、多图冲突、常识补写、未知类、损坏图片、超限输入和 JSON 错误。每个失败样本保存 `sample_id`、原始媒体哈希、模型和 adapter revision、prompt 版本、原始输出、解析结果、期望结果与路由原因。

面试时重点解释：为什么目录元数据不是视觉真值；为什么按商品而不是按图片切分；为什么 CIDEr 不能当商品字段主指标；怎样证明收益来自 SFT；为什么 unknown 与人工复核不是“模型失败”，而是系统可信度设计的一部分。

当前简历只能写已经交付的数据合同、校验器、评测器、训练配置和路由设计。不能写“CIDEr 提升 12%”“吞吐提升 2.8 倍”“显存下降 65%”之类没有实验制品支持的数字。

## 10. 项目交付

不安装 GPU 依赖也能完成脚手架验证：

```powershell
python -m unittest discover -s tests -v
python scripts/validate_data.py data/sample/train.jsonl
python scripts/validate_data.py data/sample/eval_sft.jsonl
python scripts/evaluate_predictions.py data/sample/eval_gold.jsonl data/sample/eval_predictions.jsonl
python -m compileall -q src scripts tests
```

主要文件：

- `data/dataset_info.json`：LLaMA-Factory 实际数据注册表。
- `configs/qwen3vl_lora_sft.yaml`：LoRA 训练入口。
- `configs/reproducibility.json`：运行前必须固定的版本与许可证快照。
- `src/vlm_product/contracts.py`：训练记录与输出合同。
- `src/vlm_product/provenance.py`：媒体 SHA-256 校验。
- `src/vlm_product/qwen3vl_adapter.py`：官方 Transformers 路径的 Zero-Shot 适配器。
- `src/vlm_product/evaluation.py`：字段、无依据属性和证据指标。
- `src/vlm_product/service.py`：接受、复核、拒绝路由。
- `configs/vllm_serve.sh`：固定 revision、adapter 与多模态限额的服务命令。
- `scripts/validate_local.cmd`：Windows 本地完整验证入口。

## 资料

- Qwen3-VL：https://github.com/QwenLM/Qwen3-VL
- Qwen3-VL-4B-Instruct 模型卡：https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct
- LLaMA-Factory：https://github.com/hiyouga/LlamaFactory
- vLLM 多模态与 LoRA：https://docs.vllm.ai/
- ABO 数据页：https://amazon-berkeley-objects.s3.amazonaws.com/index.html
- AWS ABO Registry：https://registry.opendata.aws/amazon-berkeley-objects/
- COCO：https://cocodataset.org/
- TextVQA：https://textvqa.org/
- A-OKVQA：https://github.com/allenai/aokvqa
- POPE：https://aclanthology.org/2023.emnlp-main.20/
