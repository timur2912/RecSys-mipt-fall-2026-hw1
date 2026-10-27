# Рекомендательные системы — домашнее задание 1

Выполнил Гаттаров Тимур, Б05-305, [Telegram](https://t.me/sugartatar)

## Датасет

**Avito ML Cup 2025** — [официальное соревнование](https://ods.ai/competitions/avitotechmlchallenge2025), данные скачиваются с [зеркала на Kaggle](https://www.kaggle.com/datasets/sharthz23/avito-cup-2025-recsys) (CC BY-NC-SA 4.0).

## Скачать данные

```bash
uv sync
uv run python recsys/download_data.py
uv run python recsys/prepare_data.py
```

## Модели

| Файл | Модель |
|---|---|
| `models/item_knn.py` | Item-based KNN, косинусное сходство |
| `models/als.py` | Implicit ALS |
| `models/ease.py` | EASE |
| `models/fm.py` | Факторизационная машина с метапризнаками объектов (категория, локация, корзина числа объявлений) |
| `models/custom_target_als.py` | ALS с кастомным таргетом: `clip((rating - 3) / 2, 0, 1) * exp(-ln(2) * age_days / half_life)` |
| `models/recommend.py` | Общая выдача top-N с исключением уже просмотренных объектов |

