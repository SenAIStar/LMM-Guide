<div align="center">
  <img src="assets/xiaosen-ai-logo.png" width="128" alt="小森学AI Logo" />

# LMM-Guide

从视觉表征到理解与生成，面向视觉多模态算法岗的学习、面试与项目代码

</div>

## 关于这个仓库

视觉多模态不只是记住 CLIP、LLaVA 或 Diffusion 的结构。真正做项目时，还要回答图像如何编码、视觉与语言如何对齐、训练数据怎样组织、生成结果如何评估，以及模型出错时应该先查数据、视觉编码器、连接层还是语言模型。

这个仓库围绕这些实际问题整理代码，内容分为讲义、面试题和项目三部分。既可以按顺序学习，也可以直接从某个项目入手，再回到对应知识点补基础。

## 内容结构

### 1. LectureNotes：大模型讲义代码

[LectureNotes](LectureNotes/) 对应视觉多模态讲义中的核心代码，主要包括：

- 视觉基础：CNN、ViT、DeiT、Swin Transformer、BEiT、U-Net、MAE、SAM；
- 视觉语言模型：CLIP、BLIP、LLaVA、MiniGPT、Qwen-VL、InternVL、DeepSeek-VL；
- 图像生成：GAN、VAE、VQ-VAE、Flow、Diffusion、Stable Diffusion、FLUX；
- 可控生成与视频生成：ControlNet、IP-Adapter、视频扩散模型；
- 统一理解与生成：Janus、Omni 等模型的关键设计。

代码会优先解释张量如何流动、训练目标为什么这样写，以及不同结构之间真正影响效果和成本的差异。

### 2. InterviewQA：大模型面试题代码实现

[InterviewQA](InterviewQA/) 用代码拆解视觉多模态面试中常见的手写题和工程题，覆盖模型原理、架构对比、模态对齐、训练与微调、强化学习、生成推理和部署优化。

这部分不会只给结论。每个问题都会尽量说明面试官在考什么、代码应该写到什么程度，以及回答继续被追问时需要补充哪些边界条件。

### 3. Projects：视觉多模态项目

[Projects](Projects/) 目前包含 7 个项目：

1. 多模态理解模型 SFT；
2. 多模态商品审核（SFT + GRPO）；
3. 商品理解问答（RAG + SFT + GRPO）；
4. 多模态无人机 Agent；
5. 可控图像生成（ControlNet + LoRA）；
6. 低成本图像生成服务；
7. Janus 统一理解与生成（SFT + GRPO）。

这些项目更关注算法岗面试里经常被追问的部分：数据如何构造、训练目标如何选择、评测集如何设计、失败样例怎样分析，以及关键模块怎样用代码实现。

## 怎么使用

如果你刚开始学视觉多模态，可以先看 `LectureNotes`，把视觉编码、跨模态对齐和生成模型的基础补齐，再选择一个项目完整走一遍。

如果你正在准备面试，可以先用 `InterviewQA` 检查自己的知识盲区，再结合 `Projects` 准备项目介绍、技术取舍和追问回答。项目代码不建议原样照搬，最好替换为自己真正跑过的数据、实验和分析。

## 代码说明

- 仓库以关键算法和核心流程为主，不按生产系统的完整标准搭建；
- 示例数据、模拟指标和流程验证结果不代表真实业务效果；
- 训练前请检查模型与数据集许可证，并根据显存调整模型规模、分辨率和 batch size；
- 涉及商品审核、内容生成和无人机控制时，需要额外考虑数据合规、生成安全与真实设备边界；
- `LectureNotes` 和 `InterviewQA` 的代码会持续补充，`Projects` 中已经提供 7 个项目的核心实现。

## 关于作者

小森，现任互联网大厂大模型算法工程师，曾在微软亚洲研究院从事算法研究工作。长期关注大模型的技术演进与工程实践，内容涉及 LLM、VLM、Diffusion、Audio、Omni 及搜索推荐等方向。

全网同名：**小森学AI**

- 小红书：[小森学AI](https://www.xiaohongshu.com/user/profile/5c5bb6f8000000001b0177fa)
- Bilibili：[小森学AI](https://space.bilibili.com/498993077)

