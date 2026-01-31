# eval_stage1_rsicd.py
import os
import random
import torch
from torch.utils.data import Subset, DataLoader

from train_stage1_rsicd import Stage1ProjectorAlignModel  # 复用你训练脚本里的模型定义
from rsicd_dataset import RSICDStage1Dataset, Stage1Collator


@torch.no_grad()
def evaluate_n_samples(model, dataset, n=8, seed=123):
    random.seed(seed)
    idxs = random.sample(range(len(dataset)), k=min(n, len(dataset)))
    subset = Subset(dataset, idxs)

    loader = DataLoader(
        subset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=Stage1Collator(pad_token_id=model.tokenizer.pad_token_id),
    )

    model.eval()
    losses = []
    for i, batch in enumerate(loader):
        # move to cuda
        for k in batch:
            batch[k] = batch[k].cuda() if torch.cuda.is_available() else batch[k]

        out = model(**batch)
        loss = out.loss.detach().float().item()
        losses.append(loss)
        print(f"[{i}] loss = {loss:.4f}")

    avg = sum(losses) / len(losses) if losses else float("nan")
    print(f"\nAvg loss over {len(losses)} samples: {avg:.4f}")
    return avg


def main():
    # ====== 路径与你训练保持一致 ======
    rsicd_root = "/mnt/data/xingyueao/BGM_IL/data/RSICD/RSICD"
    ann_path = os.path.join(rsicd_root, "dataset_rsicd.json")
    images_dir = os.path.join(rsicd_root, "RSICD_images")

    qwen_path = "/mnt/data/zhuxiang/Qwen/Qwen3-4B"
    remoteclip_ckpt = "/home/xingyueao/RemoteClip/chendelong/RemoteClip/RemoteCLIP-ViT-L-14.pt"

    # 你训练输出的 projector（改成你的实际路径）
    projector_path = "/mnt/data/zhuxiang/Qwen/Remote-Clip+Qwen/outputs/stage1_rsicd_projector/projector.pt"

    # ====== build model ======
    model = Stage1ProjectorAlignModel(
        qwen_path=qwen_path,
        remoteclip_ckpt_path=remoteclip_ckpt,
        image_token="<image>",
        trust_remote_code=False,
    )
    if torch.cuda.is_available():
        model = model.cuda()

    dataset = RSICDStage1Dataset(
        ann_path=ann_path,
        images_dir=images_dir,
        preprocess=model.vision_tower.preprocess,
        tokenizer=model.tokenizer,
        image_token="<image>",
        max_length=256,
    )

    print("\n=== (A) Evaluate BEFORE loading trained projector ===")
    evaluate_n_samples(model, dataset, n=8, seed=123)

    if os.path.isfile(projector_path):
        print("\nLoading trained projector from:", projector_path)
        state = torch.load(projector_path, map_location="cpu")
        model.vision_tower.projector.load_state_dict(state, strict=True)

        print("\n=== (B) Evaluate AFTER loading trained projector ===")
        evaluate_n_samples(model, dataset, n=8, seed=123)
    else:
        print("\n[WARN] projector.pt not found. Skipping AFTER evaluation.")
        print("Expected at:", projector_path)


if __name__ == "__main__":
    main()
