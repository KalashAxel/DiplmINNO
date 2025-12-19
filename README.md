# Проект: вариант В (кредитный скоринг)

## Что внутри
- notebooks/ — 4 ноутбука (подготовка данных, базовая модель, эксперименты, итоговая оценка).
- sql/ — schema.sql, layers.sql, queries.sql для PostgreSQL.
- src/utils.py — вспомогательные функции (метрики, порог, недискриминационность, скорость).

## Как запустить
1) Положи CSV в папку data/ как `GiveMeSomeCredit-training.csv`.
2) Открой Jupyter из корня проекта.
3) Запускай ноутбуки по порядку.

## PostgreSQL
- Выполни sql/schema.sql
- Импортируй CSV в raw.credit (через COPY/pgAdmin)
- Выполни sql/layers.sql
- Используй sql/queries.sql для контроля качества.
