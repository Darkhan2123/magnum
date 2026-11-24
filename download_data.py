# скачивает датасет с kaggle через kagglehub
# датасет публичный, логин обычно не нужен
# путь к данным потом передаётся в скрипты через --data_dir или переменную VEG_DATA
import kagglehub

path = kagglehub.dataset_download("preethi866/vegetable-dataset")
print("dataset path:", path)
print("дальше можно сделать так (PowerShell):")
print(f'$env:VEG_DATA = "{path}"')
