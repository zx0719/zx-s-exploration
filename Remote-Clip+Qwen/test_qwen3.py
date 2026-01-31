import torch
from PIL import Image
from Qwen3RemoteVLM import Qwen3RemoteVLM


def main():
    # 1) build model
    model = Qwen3RemoteVLM(
        qwen_name_or_path="/mnt/data/zhuxiang/Qwen/Qwen3-4B",
        remoteclip_ckpt_path="/home/xingyueao/RemoteClip/chendelong/RemoteClip/RemoteCLIP-ViT-L-14.pt",
        image_token="<image>",
        device="cuda" if torch.cuda.is_available() else "cpu",
        trust_remote_code=False,
    )

    # 2) preprocess image (用你 RemoteVisionTower 内的 preprocess)
    # img = Image.open("/home/xingyueao/RemoteClip/assets/airport.jpg").convert("RGB")
    img = Image.open("/home/zhuxiang/Qwen/Remote-Clip+Qwen/test.jpg").convert("RGB")
    pixel_values = model.vision_tower.preprocess(img).unsqueeze(0)  # (1,3,224,224)

    # 3) prompt 必须包含 <image>
    # prompt = "你是遥感解译助手。请描述这张图像内容：<image>\n回答："
    prompt = "请描述这张图像内容：<image>\n回答："

    # 4) generate
    out = model.generate(
        image=pixel_values,
        prompt=prompt,
        max_new_tokens=80 * 2,
        do_sample=False,
    )


    print("=== OUTPUT ===")
    print(out)


if __name__ == "__main__":
    main()
