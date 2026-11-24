# запуск: python predict.py --input путь/к/папке --output preds.csv
# работает на CPU и без интернета (веса берутся из weights/)
import argparse
import csv
import os
import sys

import torch

from common import CLASSES, IMG_EXTS, build_model, eval_tf, load_image

parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True, help="папка с фото")
parser.add_argument("--output", default="preds.csv")
parser.add_argument("--weights", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights", "resnet18_veg.pth"))
parser.add_argument("--batch_size", type=int, default=16)
args = parser.parse_args()

device = torch.device("cpu")
model = build_model(pretrained=False)
state = torch.load(args.weights, map_location=device)
model.load_state_dict(state)
model.eval()

# ищем картинки и в подпапках тоже, в csv пишем путь относительно --input
files = []
for root, _, names in os.walk(args.input):
    for f in names:
        if f.lower().endswith(IMG_EXTS):
            files.append(os.path.relpath(os.path.join(root, f), args.input).replace("\\", "/"))
files.sort()
print(f"found {len(files)} images in {args.input}")

rows = []
batch_x, batch_names = [], []


def run_batch():
    with torch.no_grad():
        probs = torch.softmax(model(torch.stack(batch_x)), dim=1)
    conf, pred = probs.max(dim=1)
    for name, p, c in zip(batch_names, pred.tolist(), conf.tolist()):
        rows.append([name, CLASSES[p], f"{c:.4f}"])
    batch_x.clear()
    batch_names.clear()


for name in files:
    try:
        img = load_image(os.path.join(args.input, name))
    except Exception as e:
        # битый файл просто пропускаем, чтобы скрипт не падал на всей папке
        print(f"WARNING: can't read {name}: {e}", file=sys.stderr)
        continue
    batch_x.append(eval_tf(img))
    batch_names.append(name)
    if len(batch_x) == args.batch_size:
        run_batch()
if batch_x:
    run_batch()

with open(args.output, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["filename", "label", "confidence"])
    w.writerows(rows)
print(f"saved {len(rows)} predictions to {args.output}")
