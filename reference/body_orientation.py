import sys

import cv2
import torch
import torchvision.transforms as transforms

# MEBOW 내부 모듈 import를 위해 경로 추가
sys.path.insert(0, "/content/MEBOW/lib")
sys.path.insert(0, "/content/MEBOW/tools")

from types import SimpleNamespace

from config import cfg, update_config


def load_mebow_model(
    cfg_path="/content/MEBOW/experiments/coco/segm-4_lr1e-3.yaml",
    img_path_dummy="/content/0322_c3_f5839949.jpg",
    device=None,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    args = SimpleNamespace(
        cfg=cfg_path,
        opts=[],
        modelDir="",
        logDir="",
        dataDir="",
        prevModelDir="",
        device=device,
        img_path=img_path_dummy,
    )

    update_config(cfg, args)

    model = eval("models." + cfg.MODEL.NAME + ".get_pose_net")(cfg, is_train=False)
    model = model.to(device)

    state_dict = torch.load(cfg.TEST.MODEL_FILE, map_location=torch.device(device))
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    return model, device


def predict_orientation(model, image_path, device):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    transform = transforms.Compose([transforms.ToTensor(), normalize])

    img = cv2.imread(image_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if img is None:
        raise FileNotFoundError(f"이미지를 찾을 수 없습니다: {image_path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (192, 256))  # 공식 demo와 동일

    x = transform(img).unsqueeze(0).float().to(device)

    with torch.no_grad():
        _, hoe_output = model(x)
        angle = int(torch.argmax(hoe_output[0]).item() * 5)

    return angle


model, device = load_mebow_model()
angle = predict_orientation(model, "/content/0324_c2_f15299705.jpg", device)
print("예측 각도:", angle)
