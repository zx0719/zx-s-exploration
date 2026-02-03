**概述**

本目录（V1）包含用于 Stage1 训练的代码：把 RemoteVisionTower 的 projector 输出与 Qwen 的文字 embedding 对齐，同时保证 LLM 回答流畅。

**文件说明**

- [train_stage1_rsicd.py](train_stage1/V1/train_stage1_rsicd.py)：训练入口，定义 `Stage1ProjectorAlignModel`，训练循环由 `transformers.Trainer` 执行。
- [rsicd_dataset.py](train_stage1/V1/rsicd_dataset.py)：数据集与 collator，实现 RSICD 数据读取与样本构造（prompt + caption）。
- [Qwen3RemoteVLM.py](train_stage1/V1/Qwen3RemoteVLM.py)：推理用示例模型（把视觉 token 插入 prompt 并调用 LLM 生成）。

**损失函数（当前实现）**

总损失：
$$\mathcal{L}=\mathcal{L}_{\mathrm{LM}}+\alpha\;\mathcal{L}_{\mathrm{align}}$$

- 语言模型损失 $\mathcal{L}_{\mathrm{LM}}$：仅对 caption token 计算（代码中通过 labels 中 `-100` 忽略 prompt 部分），使用 LLM 返回的 `loss`。 
- 对齐损失 $\mathcal{L}_{\mathrm{align}}$：批内对比（InfoNCE），将视觉向量（projector 输出按空间平均）与 caption 的文字 embedding（labels != -100 的 token embedding 的均值）作为正样本对。归一化后计算相似度矩阵并除以温度 $\tau$，使用交叉熵：

$$v_b=\frac{1}{T_v}\sum_i v_{b,i},\quad t_b=\frac{1}{|C_b|}\sum_{t\in C_b} e_{b,t}$$
$$\tilde v_b=\frac{v_b}{\|v_b\|},\quad \tilde t_b=\frac{t_b}{\|t_b\|}$$
$$s_{b,k}=\frac{\tilde v_b^\top\tilde t_k}{\tau}$$
$$\mathcal{L}_{\mathrm{align}}=-\frac{1}{B}\sum_{b=1}^B\log\frac{\exp(s_{b,b})}{\sum_{k=1}^B\exp(s_{b,k})}$$

代码中的对应变量：`align_weight` = $\alpha$，`align_temp` = $\tau$。

**关键超参（代码默认）**

- `align_weight=1.0`：对齐损失权重。
- `align_temp=0.07`：对比温度。
- 训练参数请在 [train_stage1_rsicd.py](train_stage1/V1/train_stage1_rsicd.py) 中的 `TrainingArguments` 调整（batch、lr、epoch 等）。

**依赖（主要）**

- Python 3.8+
- torch
- transformers
- swanlab (可选，仅用于记录实验)
- RemoteClip: 本工程从自定义路径导入 `RemoteVisionTower`，请确保 `sys.path` 指向 RemoteClip 源码或安装 RemoteClip。

**快速 Smoke Test（本地运行前请按需修改脚本中数据路径）**

在不能直接跑完整训练时，可运行一个小规模前向检查：

```bash
python - <<'PY'
from train_stage1_rsicd import Stage1ProjectorAlignModel
import torch

model = Stage1ProjectorAlignModel(
    qwen_path='/path/to/qwen',
    remoteclip_ckpt_path='/path/to/RemoteCLIP-ViT-L-14.pt',
    align_weight=1.0,
    align_temp=0.07,
)
model.eval()

# 构造假输入（小 batch）
B = 2
pixel_values = torch.randn(B,3,224,224)
enc = model.tokenizer(['Describe the remote sensing image: <image>\nCaption: test']*B, return_tensors='pt', padding=True)
input_ids = enc['input_ids']
attention_mask = enc['attention_mask']
labels = input_ids.clone()
prompt_len = 10
labels[:,:prompt_len] = -100

out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, pixel_values=pixel_values)
print('loss:', out.loss)
PY
```

注意：如果 RemoteVisionTower 依赖文件路径不同，可能需要先修改 `sys.path` 或安装 RemoteClip。

**保存与输出**

- 训练结束后，projector 权重保存在 `outputs/stage1_rsicd_projector/projector.pt`（由 `out_dir` 决定），并保存 tokenizer。

**调试与常见问题**

- OOM：降低 `per_device_train_batch_size` 或 `gradient_accumulation_steps`。
- 对齐效果差：尝试调整 `align_weight`（如 0.1, 0.5, 2.0）和 `align_temp`（如 0.03~0.2）。
- LLM 不更新：本阶段默认冻结 LLM，仅训练 projector；若需联合微调，请解除 LLM 参数冻结（非默认）。

如需我把 README 转为英文版、添加示例命令或直接帮你运行一次 smoke test，请告诉我要用的 `qwen_path`、`remoteclip_ckpt_path` 和较小的数据样例路径。
