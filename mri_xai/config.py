import torch


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


DEVICE = get_device()

TARGET_SLICES = 30

CLASS_NAMES = ["CN", "MCI", "AD"]

LABEL_MAP = {
    "CN": 0,
    "MCI": 1,
    "AD": 2,
}

LABEL_NAMES = {
    0: "CN",
    1: "MCI",
    2: "AD",
}

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)