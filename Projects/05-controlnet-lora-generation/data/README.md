# 数据目录

生产数据不提交到仓库。先把授权目标图放到 `raw/images/`，用 `scripts/build_canny.py` 生成条件图，再生成最终 `manifest.jsonl`。最终清单的每一行都必须包含：

- `sample_id`：全局唯一样本 ID。
- `image`、`conditioning_image`：相对数据根目录的路径，尺寸必须一致。
- `text`：描述目标内容，不把水印、商标或未授权人名写成训练触发词。
- `subject_id`、`capture_group`：用于按商品主体或同一拍摄批次分组切分。
- `license_id`：指向外部授权台账；公开可访问不等于已授权训练。
- `split`：`train`、`validation` 或 `test`，同一 `capture_group` 只能属于一个集合。
- `condition_type`：`canny`、`depth`、`pose` 或 `segmentation`。
- `sha256_image`、`sha256_conditioning`：冻结文件内容，防止训练后数据被静默替换。

`sample/` 只用于验证仓库代码，不代表真实训练数据，也不能用于报告模型效果。
