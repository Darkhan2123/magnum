# общие вещи для train.py / evaluate.py / predict.py
import os
import random

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageOps
from torchvision import models, transforms

CLASSES = ["Capsicum", "Carrot", "Cucumber", "Potato", "Raddish", "Squash"]
IMG_SIZE = 224
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# нормализация как у imagenet, потому что сеть предобучена на нём
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def get_data_dir(arg=None):
    # порядок: аргумент --data_dir, потом переменная окружения VEG_DATA, потом папка data/
    if arg:
        return arg
    return os.environ.get("VEG_DATA", "data")


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_image(path):
    img = Image.open(path)
    # фото с телефона часто повернуты через EXIF, без этого картинка может быть боком
    img = ImageOps.exif_transpose(img)
    return img.convert("RGB")


# аугментации для обучения (почему такие - в отчёте)
train_tf = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.6, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.02),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

# для валидации/теста/predict без случайностей.
# не сплющиваю в квадрат: фото в датасете 2084x4624, овощ бы сильно исказился.
# меньшая сторона -> 224 и центральный квадрат (кадры 200x200 остаются целиком)
eval_tf = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


def build_model(pretrained=True):
    if pretrained:
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    else:
        # для predict.py: веса грузим из файла, интернет не нужен
        model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
    return model


class ImageDataset(torch.utils.data.Dataset):
    # paths - список путей, labels - список индексов классов
    def __init__(self, paths, labels, tf):
        self.paths = list(paths)
        self.labels = list(labels)
        self.tf = tf

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = load_image(self.paths[i])
        return self.tf(img), self.labels[i]
