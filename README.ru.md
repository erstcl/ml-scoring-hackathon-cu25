[🇷🇺 Русская версия](README.ru.md) | [🇬🇧 English version](README.md)

---

# T-Bank Credit Attrition Prediction

[![Notebook quality](https://github.com/erstcl/ml-scoring-hackathon-cu25/actions/workflows/notebook-quality.yml/badge.svg)](https://github.com/erstcl/ml-scoring-hackathon-cu25/actions/workflows/notebook-quality.yml)

**Соревнование**: CU 2025 Scoring
**Platform**: Kaggle  
**Organizer**: Central University & T-Bank  
**Timeline**: November 2025

---

## Задача

Разработать модель машинного обучения для предсказания досрочного погашения кредита (attrition) на основе датасета кредитных продуктов T-Банка. Когда клиент выплачивает кредит досрочно, банк получает меньше процентных доходов — важно предсказывать это на этапе одобрения заявки.

---

## Особенности данных

### Ключевые характеристики
- **Целевая переменная**: `a6_flg` (флаг досрочного погашения)
- **Продукты**: 4 кредитных продукта (product_1 — product_4)
- **Временной период**: данные разбиты по месяцам (`month_dt`)
- **Признаки**: ~100+ фичей (feature_0 — feature_N)

### Главный вызов
**Temporal distribution shift**: тестовая выборка существенно отличается от тренировочной по временному распределению. Требуется контроль стабильности модели по месяцам и борьба с переобучением.

---

## Решение

### Анализ данных (EDA)

**Пропущенные значения**:
- Обнаружено множество признаков с высоким процентом пропусков (50%+, 70%+)
- Проведен анализ важности признаков с пропусками с помощью RandomForest
- Удалены признаки с >70% пропусков
- Для остальных признаков применена медианная импутация (`SimpleImputer`)

**Дисбаланс классов**:
- Целевая переменная имеет дисбаланс (attrition — более редкое событие)
- Использована стратифицированная валидация для сохранения пропорций классов

### Моделирование

**Выбор модели**: CatBoostClassifier

**Причины выбора**:
- Нативная работа с категориальными признаками
- Устойчивость к переобучению (Ordered Boosting)
- Эффективность на табличных данных
- Встроенная обработка пропусков

**Гиперпараметры**:
```python
CatBoostClassifier(
    iterations=700,
    learning_rate=0.03,
    depth=6,
    eval_metric='AUC',
    random_state=42
)
```

**Валидация**:
- Train-test split с `stratify=y` (80/20)
- Мониторинг ROC-AUC на валидационной выборке
- Early stopping с `use_best_model=True`

Random stratified split использовался для быстрой итерации модели, но не имитирует
временное распределение competition test. Kaggle leaderboard score служит внешней
оценкой при этом сдвиге.

---

## Результаты

### Метрики
- **Baseline**: 0.73707 ROC-AUC
- **Kaggle leaderboard**: 45 место, score: 0.75046

### Выводы
- Удаление сильно разреженных признаков (>70% пропусков) улучшило стабильность модели
- Public leaderboard score остался выше локального baseline при различии
  распределений train и test
- Медианная импутация оказалась эффективной для числовых признаков со средним уровнем пропусков

---

## Воспроизведение анализа

```bash
git clone https://github.com/erstcl/ml-scoring-hackathon-cu25.git
cd ml-scoring-hackathon-cu25
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
jupyter lab kaggle_hackathon.ipynb
```

При запуске нужно авторизоваться в Kaggle и подключить данные соревнования
`cu-2025-scoring`. CI проверяет структуру и сохранённые outputs notebook без
публикации competition files.

## Технологии

### ML Stack
- **Data processing**: `pandas`, `numpy`
- **Visualization**: `matplotlib`
- **Modeling**: `CatBoost`, `RandomForestClassifier` (для feature importance)
- **Metrics**: `roc_auc_score`

### Environment
- **Platform**: Kaggle Notebooks (NVIDIA Tesla T4 GPU)
- **Language**: Python 3.11

---

## О соревновании

**Ссылка на соревнование**: [Kaggle — CU 2025 Scoring](https://www.kaggle.com/competitions/cu-2025-scoring)

Данные соревнования не распространяются в репозитории. Для воспроизведения нужен
доступ к competition files на Kaggle.
