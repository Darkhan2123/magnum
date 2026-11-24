# Распознавание овощей по фото (тестовое задание, трек A)

Классификатор на 6 классов (Capsicum, Carrot, Cucumber, Potato, Raddish, Squash) на ResNet18 с transfer learning.
Датасет: [kaggle.com/datasets/preethi866/vegetable-dataset](https://www.kaggle.com/datasets/preethi866/vegetable-dataset).

Отчёт с выводами: [REPORT.md](REPORT.md). Разбор данных: [notebooks/01_data_analysis.ipynb](notebooks/01_data_analysis.ipynb).

Коротко про результат:

| проверка | accuracy | macro-F1 |
|---|---|---|
| готовое разбиение автора (test почти весь из копий train) | 1.000 | 1.000 |
| моё разбиение по дням съёмки (test) | 0.854 | 0.841 |

Готовому разбиению я не доверяю (90% test это побайтные копии train), подробности в отчёте.

## Что где лежит

```
download_data.py     скачать датасет через kagglehub
make_split.py        моё разбиение train/val/test по съёмкам -> splits/split.csv
notebooks/01_data_analysis.ipynb   часть 1: разбор данных, дубликаты, проверка разбиения
common.py            общие вещи: классы, трансформации, модель, загрузка картинки
train.py             обучение
evaluate.py          метрики, матрица ошибок, Grad-CAM, искажения, время на CPU
predict.py           предсказание для папки с фото -> csv
weights/resnet18_veg.pth   итоговые веса (~45 МБ)
splits/split.csv     моё разбиение (закоммичено, чтобы было воспроизводимо)
reports/             metrics.json, test_predictions.csv (ответ модели на каждую картинку test),
                     история обучения, картинки в reports/figures/
```

## Установка

Нужен Python 3.10-3.13 (я всё запускал на 3.12.10, Windows 11).

```
python -m venv .venv
.venv\Scripts\activate          (на linux/mac: source .venv/bin/activate)
pip install -r requirements.txt
```

На Windows из PyPI ставится CPU-версия torch, для `predict.py` этого достаточно.
На Linux из PyPI ставится версия с CUDA (она тяжёлая, но на CPU тоже работает). Если нужна именно CPU-версия:
`pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cpu`, потом `pip install -r requirements.txt`.

Для обучения на видеокарте я ставил torch с CUDA:

```
pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cu126
```

## predict.py

```
python predict.py --input путь/к/папке --output preds.csv
```

* работает на CPU и без интернета, веса берутся из `weights/resnet18_veg.pth` (можно указать другие через `--weights`)
* берёт файлы .jpg .jpeg .png .bmp .webp (регистр не важен), в том числе из подпапок; в `filename` пишется путь относительно `--input`
* учитывает поворот из EXIF (фото с телефона)
* если файл не открывается, он пропускается с предупреждением в stderr, остальные обрабатываются
* `confidence` это максимальная вероятность softmax

Формат:

```
filename,label,confidence
IMG_0001.jpg,Carrot,0.9300
```

Время на одно фото на CPU (i5-12500H): около 23 мс для маленькой картинки и около 150 мс для фото 9.6 Мп с телефона
(большая часть времени уходит на чтение JPEG, сама сеть около 21 мс; от запуска к запуску цифры плавают примерно на 10%).

## Как воспроизвести всё с нуля

1. Скачать данные (датасет публичный, ~1.7 ГБ). Скрипт напечатает путь, его надо положить в переменную `VEG_DATA`:

   ```
   python download_data.py
   $env:VEG_DATA = "C:\...\kagglehub\datasets\preethi866\vegetable-dataset\versions\1"     (PowerShell)
   export VEG_DATA=/.../vegetable-dataset/versions/1                                         (bash)
   ```

   Можно и без переменной: распаковать архив в папку `data/` (чтобы были `data/Dataset/Dataset/...` и `data/final_dataset/...`)
   или передавать `--data_dir` в каждый скрипт.

2. Разбиение (результат уже лежит в `splits/split.csv`, скрипт должен выдать такой же файл):

   ```
   python make_split.py
   ```

3. Разбор данных (часть 1). Ноутбук уже сохранён с выводами, перезапустить можно так:

   ```
   jupyter nbconvert --to notebook --execute --inplace notebooks/01_data_analysis.ipynb
   ```

4. Обучение. На RTX 3050 Ti (4 ГБ): на моём разбиении около 4 минут, на разбиении автора около 7 минут.

   ```
   python train.py                  # моё разбиение -> weights/resnet18_veg.pth
   python train.py --split author   # разбиение автора, только для сравнения -> weights/resnet18_author_split.pth
   ```

5. Оценка (метрики, матрицы ошибок, Grad-CAM, искажения, время на CPU) -> `reports/metrics.json` и `reports/figures/`:

   ```
   python evaluate.py
   ```

6. Эксперимент без весов классов (в отчёте есть сравнение):

   ```
   python train.py --no_class_weights     # -> weights/resnet18_no_class_weights.pth
   python evaluate.py --weights weights/resnet18_no_class_weights.pth --only_metrics --out reports/metrics_no_class_weights.json
   ```

Сиды зафиксированы (42), но на GPU результаты могут немного отличаться от запуска к запуску из-за cuDNN.
В репозитории лежат только итоговые веса `resnet18_veg.pth`, остальные модели нужны только для сравнения и обучаются командами выше.

## Версии

Все цифры получены на: Python 3.12.10, torch 2.14.1, torchvision 0.29.1, numpy 2.5.4, pandas 3.1.1, pillow 13.0.0,
scikit-learn 1.9.1, matplotlib 3.11.2, ImageHash 4.3.2, kagglehub 1.0.3, jupyter 1.1.1, nbconvert 7.17.2, ipykernel 7.4.0.

В `requirements.txt` torch и torchvision закреплены точно, а остальное задано диапазонами (с нижней границей),
чтобы ставилось и на Python 3.10/3.11 (numpy 2.5 требует 3.12+). Установку я проверял на 3.12, на других версиях
проверял только, что pip находит подходящие пакеты (`pip install --dry-run`).
