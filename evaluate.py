# оценка модели: метрики на test, матрица ошибок, Grad-CAM, время на CPU,
# и простая проверка на искажения (как грубая замена фото из магазина)
#   python evaluate.py
# результаты: reports/metrics.json и картинки в reports/figures/
import argparse
import io
import json
import os
import platform
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image, ImageEnhance, ImageFilter
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torchvision import transforms

from common import CLASSES, IMG_SIZE, MEAN, STD, ImageDataset, build_model, eval_tf, get_data_dir, load_image
from train import load_lists, predict

FIG = "reports/figures"


def get_metrics(y, pred):
    rep = classification_report(y, pred, labels=range(len(CLASSES)), target_names=CLASSES, output_dict=True, zero_division=0)
    return {
        "n": int(len(y)),
        "accuracy": round(float(accuracy_score(y, pred)), 4),
        "macro_f1": round(float(f1_score(y, pred, average="macro", labels=range(len(CLASSES)), zero_division=0)), 4),
        "per_class": {c: {k: round(float(rep[c][k]), 4) for k in ["precision", "recall", "f1-score", "support"]} for c in CLASSES},
    }


def plot_cm(y, pred, title, path):
    cm = confusion_matrix(y, pred, labels=range(len(CLASSES)))
    if path is None:  # только матрица, без картинки
        return cm.tolist()
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, cmap="Blues")
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            plt.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.xticks(range(len(CLASSES)), CLASSES, rotation=45)
    plt.yticks(range(len(CLASSES)), CLASSES)
    plt.xlabel("predicted"); plt.ylabel("true"); plt.title(title)
    plt.tight_layout(); plt.savefig(path, dpi=100); plt.close()
    return cm.tolist()


# ---------- искажения ----------
center = transforms.Compose([transforms.Resize(IMG_SIZE), transforms.CenterCrop(IMG_SIZE)])
to_tensor = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])


def darker(img):
    return ImageEnhance.Brightness(img).enhance(0.5)


def warm_light(img):
    # жёлтый свет ламп: больше красного, меньше синего
    r, g, b = img.split()
    return Image.merge("RGB", (r.point(lambda v: min(255, int(v * 1.15))), g, b.point(lambda v: int(v * 0.75))))


def blur(img):
    return img.filter(ImageFilter.GaussianBlur(3))


def jpeg(img):
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=15)
    return Image.open(io.BytesIO(buf.getvalue())).convert("RGB")


def zoom_out(img):
    # овощ в 2 раза меньше на светлом сером фоне (как товар на полке издалека)
    canvas = Image.new("RGB", (IMG_SIZE, IMG_SIZE), (200, 200, 195))
    small = img.resize((IMG_SIZE // 2, IMG_SIZE // 2))
    canvas.paste(small, (IMG_SIZE // 4, IMG_SIZE // 4))
    return canvas


PERTURB = {"clean": None, "darker": darker, "warm_light": warm_light, "blur": blur, "jpeg_q15": jpeg, "zoom_out": zoom_out}


class PerturbDataset(torch.utils.data.Dataset):
    def __init__(self, paths, labels, fn):
        self.paths, self.labels, self.fn = paths, labels, fn

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = center(load_image(self.paths[i]))
        if self.fn is not None:
            img = self.fn(img)
        return to_tensor(img), self.labels[i]


# ---------- Grad-CAM ----------
class GradCAM:
    # Grad-CAM руками: сохраняю активации последнего блока и градиенты по ним
    def __init__(self, model, layer):
        self.model = model
        self.acts = None
        self.grads = None
        layer.register_forward_hook(self.hook)

    def hook(self, module, inp, out):
        self.acts = out.detach()
        out.register_hook(self.save_grad)

    def save_grad(self, grad):
        self.grads = grad.detach()

    def __call__(self, x, class_idx):
        self.model.zero_grad()
        out = self.model(x)
        out[0, class_idx].backward()
        weights = self.grads.mean(dim=(2, 3), keepdim=True)  # среднее градиента по каждому каналу
        cam = F.relu((weights * self.acts).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False)[0, 0]
        cam = cam / (cam.max() + 1e-8)
        return cam.cpu().numpy()


def gradcam_grid(model, device, paths, labels, preds, probs, idx_list, path):
    cammer = GradCAM(model, model.layer4)
    n = len(idx_list)
    fig, axes = plt.subplots(2, (n + 1) // 2, figsize=(3.2 * ((n + 1) // 2), 7))
    axes = axes.ravel()
    for ax, i in zip(axes, idx_list):
        img = center(load_image(paths[i]))
        x = to_tensor(img).unsqueeze(0).to(device)
        cam = cammer(x, int(preds[i]))
        ax.imshow(img)
        ax.imshow(cam, cmap="jet", alpha=0.45)
        ok = "OK" if preds[i] == labels[i] else "ОШИБКА"
        ax.set_title(f"{ok}\ntrue {CLASSES[labels[i]]}\npred {CLASSES[preds[i]]} {probs[i].max():.2f}", fontsize=9,
                     color="green" if preds[i] == labels[i] else "red")
        ax.axis("off")
    for ax in axes[n:]:
        ax.axis("off")
    plt.tight_layout(); plt.savefig(path, dpi=90); plt.close()


# ---------- время на CPU ----------
def cpu_timing(weights, frame_paths, photo_paths):
    torch.set_grad_enabled(False)
    t0 = time.perf_counter()
    model = build_model(pretrained=False)
    model.load_state_dict(torch.load(weights, map_location="cpu"))
    model.eval()
    load_sec = time.perf_counter() - t0

    x = eval_tf(load_image(frame_paths[0])).unsqueeze(0)
    for _ in range(5):  # прогрев
        model(x)

    res = {"model_load_sec": round(load_sec, 3), "torch_threads": torch.get_num_threads(),
           "cpu": platform.processor(), "torch": torch.__version__}
    for name, paths in [("frames_200x200", frame_paths), ("full_photos", photo_paths)]:
        total, fwd = [], []
        for p in paths:
            t0 = time.perf_counter()
            x = eval_tf(load_image(p)).unsqueeze(0)
            t1 = time.perf_counter()
            model(x)
            t2 = time.perf_counter()
            total.append(t2 - t0)
            fwd.append(t2 - t1)
        res[name] = {"n": len(paths), "total_ms_mean": round(1000 * float(np.mean(total)), 1),
                     "total_ms_median": round(1000 * float(np.median(total)), 1),
                     "model_only_ms_mean": round(1000 * float(np.mean(fwd)), 1)}
    torch.set_grad_enabled(True)
    return res


def plot_history():
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    for split in ["my", "author"]:
        p = f"reports/history_{split}.json"
        if not os.path.exists(p):
            continue
        h = pd.DataFrame(json.load(open(p))["history"])
        axes[0].plot(h.epoch, h.train_loss, marker="o", label=f"train loss ({split})")
        axes[1].plot(h.epoch, h.train_acc, marker="o", label=f"train acc ({split})")
        axes[1].plot(h.epoch, h.val_macro_f1, marker="s", label=f"val macro-F1 ({split})")
    for ax in axes:
        ax.set_xlabel("epoch"); ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(f"{FIG}/training_curves.png", dpi=100); plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--split_csv", default="splits/split.csv")
    parser.add_argument("--weights", default="weights/resnet18_veg.pth")
    parser.add_argument("--author_weights", default="weights/resnet18_author_split.pth")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--out", default="reports/metrics.json")
    parser.add_argument("--only_metrics", action="store_true", help="только метрики на моём разбиении (для экспериментов)")
    args = parser.parse_args()

    os.makedirs(FIG, exist_ok=True)
    data_dir = get_data_dir(args.data_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = {}

    # 1. моё разбиение
    lists = load_lists("my", data_dir, args.split_csv)
    split = pd.read_csv(args.split_csv)
    model = build_model(pretrained=False)
    model.load_state_dict(torch.load(args.weights, map_location="cpu"))
    model = model.to(device).eval()

    res = {}
    for part in ["val", "test"]:
        loader = torch.utils.data.DataLoader(ImageDataset(*lists[part], eval_tf), batch_size=64, num_workers=args.num_workers)
        probs, y = predict(model, loader, device)
        pred = probs.argmax(1)
        res[part] = get_metrics(y, pred)
        if part == "test":
            test_probs, test_y, test_pred = probs, y, pred
    is_frame = split.loc[split.split == "test", "is_frame"].values
    res["test_only_frames"] = get_metrics(test_y[is_frame], test_pred[is_frame])
    res["test_only_photos"] = get_metrics(test_y[~is_frame], test_pred[~is_frame])
    # для экспериментов (--only_metrics) картинку не перезаписываю
    cm_path = None if args.only_metrics else f"{FIG}/cm_my_split.png"
    res["test_confusion_matrix"] = plot_cm(test_y, test_pred, "моё разбиение, test", cm_path)

    # где именно ошибки: по дням и по видео / сериям фото
    t = split[split.split == "test"].copy()
    t["pred"] = [CLASSES[i] for i in test_pred]
    t["confidence"] = test_probs.max(1).round(4)
    t["wrong"] = t["pred"] != t["cls"]
    err = t[t["wrong"]]
    res["test_errors_by_day"] = {f"{c} {d} -> {p}": int(n) for (c, d, p), n in err.groupby(["cls", "day", "pred"]).size().items()}
    by_group = t.groupby("group")["wrong"].agg(["sum", "size"])
    by_group = by_group[by_group["sum"] > 0].sort_values("sum", ascending=False)
    res["test_errors_by_group"] = {g: f"{int(r['sum'])}/{int(r['size'])}" for g, r in by_group.iterrows()}
    res["test_errors_conf_over_0.9"] = int((err["confidence"] > 0.9).sum())
    if not args.only_metrics:
        t.to_csv("reports/test_predictions.csv", index=False)
    results["my_split"] = res
    print("my split  val :", res["val"]["accuracy"], res["val"]["macro_f1"])
    print("my split  test:", res["test"]["accuracy"], res["test"]["macro_f1"])
    print(pd.DataFrame(res["test"]["per_class"]).T)

    if args.only_metrics:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print("saved", args.out)
        return

    # 2. разбиение автора (если модель обучена: python train.py --split author)
    if os.path.exists(args.author_weights):
        a_lists = load_lists("author", data_dir, None)
        a_model = build_model(pretrained=False)
        a_model.load_state_dict(torch.load(args.author_weights, map_location="cpu"))
        a_model = a_model.to(device).eval()
        loader = torch.utils.data.DataLoader(ImageDataset(*a_lists["test"], eval_tf), batch_size=64, num_workers=args.num_workers)
        probs, y = predict(a_model, loader, device)
        results["author_split"] = {"test": get_metrics(y, probs.argmax(1))}
        results["author_split"]["test_confusion_matrix"] = plot_cm(y, probs.argmax(1), "разбиение автора, test", f"{FIG}/cm_author_split.png")
        print("author split test:", results["author_split"]["test"]["accuracy"], results["author_split"]["test"]["macro_f1"])

    # 3. искажения на моём test
    rob = {}
    for name, fn in PERTURB.items():
        loader = torch.utils.data.DataLoader(PerturbDataset(*lists["test"], fn), batch_size=64, num_workers=args.num_workers)
        probs, y = predict(model, loader, device)
        m = get_metrics(y, probs.argmax(1))
        rob[name] = {"accuracy": m["accuracy"], "macro_f1": m["macro_f1"], "mean_confidence": round(float(probs.max(1).mean()), 4)}
        print("perturb", name, rob[name])
    results["robustness_test"] = rob
    # пример искажений для отчёта
    ex = center(load_image(lists["test"][0][0]))
    fig, axes = plt.subplots(1, len(PERTURB), figsize=(2.5 * len(PERTURB), 2.8))
    for ax, (name, fn) in zip(axes, PERTURB.items()):
        ax.imshow(ex if fn is None else fn(ex)); ax.set_title(name); ax.axis("off")
    plt.tight_layout(); plt.savefig(f"{FIG}/perturbations.png", dpi=80); plt.close()

    # 4. Grad-CAM: до 5 ошибок + правильные, чтобы всего было 10
    rng = np.random.RandomState(0)
    test_paths = lists["test"][0]
    wrong = np.where(test_pred != test_y)[0]
    right = np.where(test_pred == test_y)[0]
    # ошибки беру из разных классов, если получится
    wrong_pick = []
    for c in rng.permutation(len(CLASSES)):
        cand = wrong[test_y[wrong] == c]
        if len(cand) and len(wrong_pick) < 5:
            wrong_pick.append(rng.choice(cand))
    rest = [i for i in rng.permutation(wrong) if i not in wrong_pick]
    wrong_pick += rest[:5 - len(wrong_pick)]
    right_pick = []
    for c in range(len(CLASSES)):
        cand = right[test_y[right] == c]
        # чередую кадры и фото, чтобы были оба типа
        want_photo = c % 2 == 1 and (~is_frame[cand]).any()
        cand = cand[~is_frame[cand]] if want_photo else cand
        if len(cand):
            right_pick.append(rng.choice(cand))
    pick = list(wrong_pick) + right_pick[:10 - len(wrong_pick)]
    gradcam_grid(model, device, test_paths, test_y, test_pred, test_probs, pick, f"{FIG}/gradcam.png")
    results["gradcam_examples"] = [{"file": os.path.basename(test_paths[i]), "true": CLASSES[test_y[i]],
                                    "pred": CLASSES[test_pred[i]], "conf": round(float(test_probs[i].max()), 3)} for i in pick]

    # 5. время на CPU
    test_split = split[split.split == "test"]
    frames = [os.path.join(data_dir, p) for p in test_split[test_split.is_frame].sample(100, random_state=0).path]
    photos = [os.path.join(data_dir, p) for p in test_split[~test_split.is_frame].path]
    results["cpu_timing"] = cpu_timing(args.weights, frames, photos)
    print("cpu timing:", results["cpu_timing"])

    plot_history()
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("saved", args.out)


if __name__ == "__main__":
    main()
