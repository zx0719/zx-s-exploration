
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

    # 2) load trained projector
    # projector_path = "/mnt/data/zhuxiang/Qwen/Remote-Clip+Qwen/outputs/V1/stage1_rsicd_projector/projector.pt"
    projector_path =  "/mnt/data/zhuxiang/Qwen/Remote-Clip+Qwen/outputs/V2/stage1_rsicd_projector/projector.pt" #加大训练数据量和训练轮数
    state = torch.load(projector_path, map_location="cpu")
    model.vision_tower.projector.load_state_dict(state, strict=True)
    model.eval()

    # 2) preprocess image (用你 RemoteVisionTower 内的 preprocess)
    img = Image.open("/home/xingyueao/RemoteClip/assets/airport.jpg").convert("RGB")
    # img = Image.open("/home/zhuxiang/Qwen/Remote-Clip+Qwen/train_stage1/test/test1.jpg").convert("RGB")
    pixel_values = model.vision_tower.preprocess(img).unsqueeze(0)  # (1,3,224,224)

    # 3) prompt 必须包含 <image>
    # prompt = "你是遥感解译助手。请描述这张图像内容：<image>\n回答："
    # prompt = "请对<image>进行场景单标签分类，仅从以下指定类别中输出唯一匹配结果，禁止新增 / 修改类别、禁止额外解释：Airport、Bare Land、Baseball Field、Beach、Bridge、Center、Church、Commercial、Dense Residential、Desert、Farmland、Forest、Industrial、Meadow、Medium Residential、Mountain、Park、School、Square、Parking、Playground、Pond、Viaduct、Port、Railway Station、Resort、River、Sparse Residential、Storage Tanks、Stadium"
    prompt = "请说明一下这个图片的场景：<image>\n回答："

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
