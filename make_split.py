# делает своё разбиение train/val/test -> splits/split.csv
# почему не готовое разбиение автора - см. notebooks/01_data_analysis.ipynb и REPORT.md
#
# идея: имена файлов это время съёмки с телефона (YYYYMMDD_HHMMSS),
# а *_frame_N.jpg это кадры из одного видео. соседние кадры почти одинаковые,
# поэтому делить надо не по картинкам, а по съёмкам.
#   test  - один день съёмки на каждый класс, который модель вообще не видит
#   val   - куски видео (по 50 кадров подряд) и серии фото из оставшихся дней.
#           val получается немного оптимистичный (соседние куски одного видео похожи),
#           он нужен только чтобы выбрать эпоху. честная оценка - test
#   train - всё остальное
import argparse
import hashlib
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from common import CLASSES, get_data_dir

# день съёмки, который уходит в тест. выбирал так, чтобы в тесте было
# хотя бы одно видео и примерно 20-40% картинок класса
TEST_DAY = {
    "Capsicum": "20251117",
    "Carrot": "20251113",
    "Cucumber": "20260101",
    "Potato": "20251113",
    "Squash": "20251113",
}
# у редиса всего один день съёмки (20251210), поэтому для него делю по времени:
# второе видео и всё, что снято после его начала, идёт в тест
RADDISH_TEST_FROM = "20251210_230730"

PHOTO_GAP_SEC = 30  # фото, снятые с разницей меньше 30 сек, считаю одной серией
CHUNK = 50  # для val режу видео на куски по 50 кадров


def md5(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def to_seconds(t):
    # "HHMMSS" -> секунды
    return int(t[:2]) * 3600 + int(t[2:4]) * 60 + int(t[4:6])


parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", default=None)
parser.add_argument("--out", default="splits/split.csv")
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

root = os.path.join(get_data_dir(args.data_dir), "Dataset", "Dataset")
rows = []
for cls in CLASSES:
    for name in sorted(os.listdir(os.path.join(root, cls))):
        rows.append({"path": f"Dataset/Dataset/{cls}/{name}", "cls": cls, "name": name})
df = pd.DataFrame(rows)
print("images:", len(df))

df["is_frame"] = df["name"].str.contains("_frame_")
df["stamp"] = df["name"].str.replace(r"(_frame_\d+)?\.jpg$", "", regex=True)  # 20251113_210729
df["day"] = df["stamp"].str[:8]

# группа: для кадров - id видео, для фото - серия фото подряд
groups = []
for (cls, day), g in df.groupby(["cls", "day"]):
    for i, r in g.iterrows():
        if r["is_frame"]:
            groups.append((i, f"{cls}_video_{r['stamp']}"))
    photos = g[~g["is_frame"]].sort_values("stamp")
    burst, prev = 0, None
    for i, r in photos.iterrows():
        t = to_seconds(r["stamp"][9:])
        if prev is not None and t - prev > PHOTO_GAP_SEC:
            burst += 1
        prev = t
        groups.append((i, f"{cls}_photos_{day}_{burst}"))
df["group"] = pd.Series(dict(groups))

# для val: целое видео слишком большое (до 607 кадров), поэтому режу на куски
frame_idx = pd.to_numeric(df["name"].str.extract(r"_frame_(\d+)\.jpg$")[0]).fillna(0).astype(int)
df["chunk"] = np.where(df["is_frame"], df["group"] + "_" + (frame_idx // CHUNK).astype(str), df["group"])

# test
is_test = np.zeros(len(df), dtype=bool)
for cls, day in TEST_DAY.items():
    is_test |= (df["cls"] == cls) & (df["day"] == day)
is_test |= (df["cls"] == "Raddish") & (df["stamp"] >= RADDISH_TEST_FROM)
df["split"] = np.where(is_test, "test", "train")

# val: из оставшегося по группам, примерно 1/5 часть
pool = df[df["split"] == "train"]
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=args.seed)
_, val_idx = next(sgkf.split(pool, pool["cls"], pool["chunk"]))
df.loc[pool.index[val_idx], "split"] = "val"

# проверки, что ничего не протекает между частями
df["md5"] = [md5(os.path.join(get_data_dir(args.data_dir), p)) for p in df["path"]]
df["is_test"] = df["split"] == "test"
assert (df.groupby("group")["is_test"].nunique() == 1).all(), "video/photo series is both in test and train"
assert (df.groupby("chunk")["split"].nunique() == 1).all(), "chunk is in several splits"
assert (df.groupby("md5")["split"].nunique() == 1).all(), "same file in several splits"
# у одного дня+класса все кадры и фото должны быть в одном куске test / не test
day_parts = df[df["cls"] != "Raddish"].groupby(["cls", "day"])["split"].apply(lambda s: set(s) == {"test"} or "test" not in set(s))
assert day_parts.all()

os.makedirs(os.path.dirname(args.out), exist_ok=True)
df[["path", "cls", "day", "group", "chunk", "is_frame", "split"]].to_csv(args.out, index=False)
print(pd.crosstab(df["cls"], df["split"], margins=True))
print("groups per split:", df.groupby("split")["group"].nunique().to_dict())
print("saved", args.out)
