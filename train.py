# обучение ResNet18 (transfer learning) на 6 классов
#   python train.py                 - на моём разбиении (splits/split.csv) -> weights/resnet18_veg.pth
#   python train.py --split author  - на готовом разбиении автора (final_dataset), только для сравнения
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, log_loss

from common import CLASSES, ImageDataset, build_model, eval_tf, get_data_dir, seed_everything, train_tf


def load_lists(split, data_dir, split_csv):
    # возвращает {"train": (paths, labels), "val": (...), "test": (...)}
    out = {}
    if split == "my":
        df = pd.read_csv(split_csv)
        for part in ["train", "val", "test"]:
            d = df[df["split"] == part]
            out[part] = ([os.path.join(data_dir, p) for p in d["path"]], [CLASSES.index(c) for c in d["cls"]])
    else:
        for part in ["train", "val", "test"]:
            paths, labels = [], []
            for c in CLASSES:
                folder = os.path.join(data_dir, "final_dataset", "final_dataset", part, c)
                for name in sorted(os.listdir(folder)):
                    paths.append(os.path.join(folder, name))
                    labels.append(CLASSES.index(c))
            out[part] = (paths, labels)
    return out


def predict(model, loader, device):
    model.eval()
    all_probs, all_y = [], []
    with torch.no_grad():
        for x, y in loader:
            probs = torch.softmax(model(x.to(device)), dim=1)
            all_probs.append(probs.cpu())
            all_y.append(y)
    return torch.cat(all_probs).numpy(), torch.cat(all_y).numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--split", default="my", choices=["my", "author"])
    parser.add_argument("--split_csv", default="splits/split.csv")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default=None)
    parser.add_argument("--no_class_weights", action="store_true", help="эксперимент: обычный loss без весов")
    args = parser.parse_args()

    if args.out is None:
        if args.split == "author":
            args.out = "weights/resnet18_author_split.pth"
        elif args.no_class_weights:
            args.out = "weights/resnet18_no_class_weights.pth"  # чтобы не затереть основные веса
        else:
            args.out = "weights/resnet18_veg.pth"
    seed_everything(args.seed)
    data_dir = get_data_dir(args.data_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    lists = load_lists(args.split, data_dir, args.split_csv)
    for part, (p, _) in lists.items():
        print(part, len(p))

    train_ds = ImageDataset(*lists["train"], train_tf)
    val_ds = ImageDataset(*lists["val"], eval_tf)
    g = torch.Generator().manual_seed(args.seed)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, generator=g,
                                               num_workers=args.num_workers, persistent_workers=args.num_workers > 0)
    # val маленький, грузим без отдельных процессов (на windows каждый воркер ест ~2 ГБ памяти)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=64, num_workers=0)

    # веса классов против дисбаланса: редкий класс весит больше
    counts = np.bincount(lists["train"][1], minlength=len(CLASSES))
    class_w = torch.tensor(counts.sum() / (len(CLASSES) * counts), dtype=torch.float32)
    print("train counts:", dict(zip(CLASSES, counts.tolist())))
    print("class weights:", [round(w, 2) for w in class_w.tolist()])

    model = build_model(pretrained=True).to(device)
    if args.no_class_weights:
        class_w = torch.ones(len(CLASSES))
    criterion = nn.CrossEntropyLoss(weight=class_w.to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    history = []
    best_f1, best_loss, best_epoch = -1, 1e9, 0
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        total_loss, correct, seen = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(y)
            correct += (out.argmax(1) == y).sum().item()
            seen += len(y)
        scheduler.step()

        probs, y_val = predict(model, val_loader, device)
        pred = probs.argmax(1)
        val_acc = accuracy_score(y_val, pred)
        val_f1 = f1_score(y_val, pred, average="macro")
        val_loss = log_loss(y_val, probs, labels=list(range(len(CLASSES))))
        row = {"epoch": epoch, "train_loss": total_loss / seen, "train_acc": correct / seen,
               "val_acc": val_acc, "val_macro_f1": val_f1, "val_loss": val_loss, "sec": time.time() - t0}
        history.append(row)
        print(f"epoch {epoch}: loss {row['train_loss']:.4f} train_acc {row['train_acc']:.4f} "
              f"val_acc {val_acc:.4f} val_f1 {val_f1:.4f} val_loss {val_loss:.4f} ({row['sec']:.0f}s)")
        # val быстро доходит до f1=1.0, поэтому при равном f1 смотрю на val loss
        if (val_f1, -val_loss) > (best_f1, -best_loss):
            best_f1, best_loss, best_epoch = val_f1, val_loss, epoch
            torch.save(model.state_dict(), args.out)
            print("  saved best ->", args.out)

    os.makedirs("reports", exist_ok=True)
    tag = args.split + ("_no_class_weights" if args.no_class_weights else "")
    with open(f"reports/history_{tag}.json", "w") as f:
        json.dump({"args": vars(args), "history": history, "best_epoch": best_epoch,
                   "best_val_macro_f1": best_f1, "best_val_loss": best_loss}, f, indent=2)
    print(f"best epoch {best_epoch}: val macro-F1 {best_f1:.4f}, val loss {best_loss:.4f}")


if __name__ == "__main__":
    main()
