# Дополнительные расчеты для доработки статьи

**Цель файла.** Этот документ нужен не для пересказа замечаний рецензентов, а как практический runbook: какие именно расчеты еще стоит добрать, чтобы закрыть оставшиеся reviewer-driven gaps в текущих версиях `paper.tex` и `response_to_reviewers.tex`.

**Что уже важно понимать.**

- Текущая версия статьи уже закрывает многие замечания текстом и существующими CSV-таблицами.
- Новые вычисления нужны только для небольшого числа замечаний, где у рецензента все еще может остаться запрос на дополнительные числа.
- В этом репозитории уже есть актуальные `paper/paper.tex` и `paper/response_to_reviewers.tex`, поэтому ниже указано, куда именно потом вставлять результаты.
- Этот файл ориентирован на автора, который сам запускает расчеты на своей машине и потом вручную обновляет статью.

---

## Какие замечания еще могут потребовать новых расчетов

| Reviewer / Comment | Что хочет увидеть рецензент | Нужен новый расчет | Приоритет |
|---|---|---|---|
| `R2-C4 / grbtune` | Понять, не сводится ли ускорение к стандартному авто-тюнингу Gurobi | Да | **Критично** |
| `R1-C6 / BANDIT vs fixed-K` | Понять, дает ли BANDIT что-то сверх простого fixed `K` | Да | **Желательно** |
| `R1-C5 / GNN P/R/F1` | Увидеть classifier-level quality, а не только downstream speedup | Формально да, но шаг сейчас заблокирован | **Опционально, сейчас заблокировано** |

**Практический вывод:** если времени мало, сначала делайте `grbtune`. Если есть еще 2-3 часа, добивайте fixed-`K` ablation. На GNN не тратьте бюджет, пока не устранен feature mismatch.

---

## Что запускать в первую очередь

### Рекомендуемый порядок

1. `grbtune` на `matpower/case118` и `matpower/case300`
2. fixed-`K` ablation (`K=64`, `128`, `256`) для `WARM+LAZY`
3. GNN P/R/F1 только если заранее починена совместимость признаков и архивных весов

### Если времени очень мало

- Минимально разумный прогон: только шаг 1 (`grbtune`).

### Если есть 3-5 часов

- Делать шаги 1 и 2.

### Если есть весь 12-часовой бюджет

- Делать шаги 1 и 2 обязательно.
- Шаг 3 делать **только** если до начала расчетов уже синхронизирован GNN feature schema.

---

## Подготовка окружения

Все команды ниже предполагают, что вы запускаете их **из корня репозитория** `/Users/gregor/Documents/SK_R/SCUCa`.

### Базовая подготовка

```bash
source venv/bin/activate
export PYTHONPATH=.
```

### Проверка базовых библиотек

```bash
python -c "import gurobipy, pandas, numpy; print('gurobi', gurobipy.gurobi.version())"
```

### Отдельная проверка для GNN

```bash
python -c "import torch, torch_geometric; print('torch', torch.__version__); print('torch_geometric', torch_geometric.__version__)"
```

Если `torch_geometric` не установлен, это **не мешает** шагам с `grbtune` и fixed-`K`.

---

## 1. `grbtune`: обязательный reviewer-facing эксперимент

### Зачем это нужно

Это ответ на `R2-C4`. Reviewer хочет понять, нельзя ли получить сопоставимое ускорение просто авто-тюнингом Gurobi, без всей learning-enhanced схемы.

### Что именно делает скрипт

Скрипт:

1. строит **обычный exact SCUC baseline model** для выбранного representative instance;
2. экспортирует его в `.mps`;
3. запускает `Model.tune()` с заданным budget;
4. сравнивает solve time default-конфигурации и tuned-конфигурации на одном и том же instance.

### Что именно сравнивается

Сравнивается:

- **default SCUC baseline**
- **tuned SCUC baseline**

Это **не** tuning ML-ускоренного варианта. Именно так и надо: reviewer спрашивает, не объясняется ли observed gain обычным solver tuning.

### Какие команды запускать

#### Обязательные

```bash
PYTHONPATH=. venv/bin/python paper/scripts/run_grbtune.py --case matpower/case118 --tune-time 1800 --solve-time 600
PYTHONPATH=. venv/bin/python paper/scripts/run_grbtune.py --case matpower/case300 --tune-time 3600 --solve-time 600
```

#### Опционально

```bash
PYTHONPATH=. venv/bin/python paper/scripts/run_grbtune.py --case matpower/case89pegase --tune-time 600 --solve-time 600
```

### Сколько это займет

- `case118`: примерно 35 минут
- `case300`: примерно 75 минут
- `case89pegase`: примерно 10-15 минут

Итого обязательный минимум: около 2 часов.

### Какие файлы появятся

Для каждого case:

- `paper/data/grbtune_<case>.json` — детальная запись tuning run
- `paper/data/grbtune_<case>.mps` — экспорт модели до tuning
- `paper/data/grbtune_<case>.prm` — лучший набор параметров Gurobi

Общая сводка:

- `paper/tables/grbtune_vs_default.csv`

### Что потом обновить в статье

После завершения этого шага нужно обновить **paragraph в subsection `Limitations and threats to validity`** в `paper/paper.tex`, а не просто “где-то в статье”.

Ориентир по месту:

- subsection `Limitations and threats to validity`
- текущий bullet `Solver tuning`

Смысл обновления:

- если tuned baseline дает только умеренный выигрыш, это усиливает ваш аргумент, что основной gain приходит от constraint management;
- если tuned baseline дает большой выигрыш, текст надо сделать осторожнее и честнее.

### Безопасная заготовка для текста статьи

После получения чисел можно подставить такую формулировку:

> On representative instances of `case118` and `case300`, Gurobi's built-in parameter tuner reduced the default solve time by approximately `X.XXx` and `Y.YYx`, respectively. This gain is smaller than the acceleration delivered by the best constraint-economy variants, which suggests that the dominant improvement comes from compact constraint handling rather than from parameter tuning alone.

---

## 2. Fixed-`K` ablation для BANDIT

### Зачем это нужно

Это ответ на `R1-C6`. Reviewer справедливо спрашивает: если BANDIT почти не лучше plain LAZY, дает ли он хоть что-то по сравнению с простыми фиксированными значениями `K`?

### Что именно нужно проверить

Надо сравнить:

- `WARM+LAZY+BANDIT`
- `WARM+LAZY+K64`
- `WARM+LAZY+K128`
- `WARM+LAZY+K256`

### Что запускать

Запускать только:

- `matpower/case118`
- `matpower/case300`

Режим:

- `WARM+LAZY`

Три отдельных прогона:

```bash
PYTHONPATH=. venv/bin/python -m src.paper.experiments \
    --cases matpower/case118 matpower/case300 \
    --modes WARM+LAZY \
    --lazy-top-k 64 \
    --time-limit 600 \
    --skip-solved \
    --train-use-existing-only

PYTHONPATH=. venv/bin/python -m src.paper.experiments \
    --cases matpower/case118 matpower/case300 \
    --modes WARM+LAZY \
    --lazy-top-k 128 \
    --time-limit 600 \
    --skip-solved \
    --train-use-existing-only

PYTHONPATH=. venv/bin/python -m src.paper.experiments \
    --cases matpower/case118 matpower/case300 \
    --modes WARM+LAZY \
    --lazy-top-k 256 \
    --time-limit 600 \
    --skip-solved \
    --train-use-existing-only
```

### Сколько это займет

Оценка по текущему проекту:

- `case118`: около 30 минут на весь sweep
- `case300`: около 2 часов на весь sweep

Итого: примерно 2.5-3 часа.

### Что важно не перепутать

`paper/scripts/postprocess.py` **не** встроит эти новые fixed-`K` результаты автоматически в старые paper-side summary tables.

Не надо обещать себе, что новые строки “сами появятся” в `speedup_with_ci.tex`. Этого в текущей инфраструктуре нет.

### Какие файлы появятся

Основные сырые результаты:

- новые CSV в `results/raw_logs/`

В них появятся режимы:

- `WARM+LAZY+K64`
- `WARM+LAZY+K128`
- `WARM+LAZY+K256`

### Как собрать сводку для статьи

После трех запусков выполните этот one-shot aggregation snippet:

```bash
PYTHONPATH=. venv/bin/python - <<'PY'
import glob
import pandas as pd

logs = sorted(glob.glob("results/raw_logs/*.csv"))
df = pd.concat((pd.read_csv(p) for p in logs), ignore_index=True)

sel = df[df["mode"].isin([
    "WARM+LAZY+K64",
    "WARM+LAZY+K128",
    "WARM+LAZY+K256",
])]

summary = (
    sel.groupby(["case_folder", "mode"])
    .agg(
        n=("runtime_sec", "size"),
        runtime_median_sec=("runtime_sec", "median"),
        runtime_mean_sec=("runtime_sec", "mean"),
        status_ok_rate=("status", lambda s: s.isin(["OPTIMAL", "SUBOPTIMAL", "TIME_LIMIT"]).mean()),
        strict_feasible_rate=("violations", lambda s: (s.astype(str) == "OK").mean()),
    )
    .reset_index()
)

print(summary.to_string(index=False))
summary.to_csv("paper/tables/lazy_topk_ablation.csv", index=False)
PY
```

### Какие числа потом нужны для текста статьи

Из `paper/tables/lazy_topk_ablation.csv` вам нужны:

- `runtime_median_sec`
- `runtime_mean_sec`
- `status_ok_rate`
- `strict_feasible_rate`

Их надо сопоставить с уже имеющимся `WARM+LAZY+BANDIT`.

### Как потом использовать это в статье

Смысл обновления:

- если лучший fixed-`K` почти не отличается от BANDIT, reviewer comment закрывается как negative result;
- если fixed-`K` лучше BANDIT, paragraph про BANDIT надо переписать еще жестче;
- если BANDIT немного лучше fixed-`K`, это уже нормальная количественная защита текущего текста.

### Безопасная формулировка для статьи

После получения чисел можно использовать такой шаблон:

> Fixed-`K` ablations on `case118` and `case300` show that the best static cut budget reaches approximately `X.XXx`, compared to `Y.YYx` for BANDIT. This confirms that the adaptive policy brings at most a marginal gain on the present benchmark.

Если `strict_feasible_rate < 1.0`, добавьте это прямо в текст, а не прячьте в логах.

---

## 3. GNN precision / recall / F1

### Статус

Этот шаг отвечает на `R1-C5`, но **в текущем snapshot он заблокирован**.

### Почему reviewer этого хочет

Reviewer просил не только downstream speedup, но и classifier-level evidence:

- precision
- recall
- возможно F1

Это логично, потому что иначе GNN section выглядит как сложная модель, которая не дает выигрыша и почти не объяснена количественно.

### Почему шаг сейчас не готов к запуску

Причина в несовместимости между архивными весами и текущим кодом:

- archived GNN weights ожидают `in_dim = 11`
- текущий extractor в `src/ml_models/gnn_screening.py` строит 7 признаков

Из-за этого запуск падает с ошибкой матричного несовпадения вида:

`mat1 and mat2 shapes cannot be multiplied`

### Быстрая проверка схемы признаков

```bash
venv/bin/python -c "
import json
from pathlib import Path
meta = Path('results/paper_upto1354_fullmodes_2026-04-17/artifacts/gnn_screening/matpower_case118/meta.json')
print(json.loads(meta.read_text())['in_dim'])
"
```

Если эта команда печатает `11`, а текущий код по-прежнему строит 7 признаков, запускать evaluation **не надо**.

### Команда запуска после починки

Только после синхронизации feature schema:

```bash
PYTHONPATH=. venv/bin/python paper/scripts/eval_gnn_precision_recall.py \
    --artifact-root results/paper_upto1354_fullmodes_2026-04-17/artifacts/gnn_screening \
    --solution-root results/paper_upto1354_fullmodes_2026-04-17/solutions/test/raw \
    --p-thr 0.60 \
    --y-thr 0.70
```

### Что появится после успешного запуска

- `paper/tables/gnn_pr_f1.csv`
- `paper/tables/gnn_pr_f1.tex`

### Практический совет

**Не тратьте на это 12-часовой бюджет без предварительной синхронизации feature schema.**

Сейчас этот шаг:

- не обязателен для submission,
- не является лучшим use of time,
- и не должен мешать более важным расчетам (`grbtune`, fixed-`K`).

---

## Как использовать результаты в статье

### Если выполнен `grbtune`

Обновить в `paper.tex`:

- subsection `Limitations and threats to validity`
- paragraph / bullet `Solver tuning`

Смысл:

- заменить чисто лимитационный текст на data-backed comparison.

### Если выполнен fixed-`K`

Обновить в `paper.tex`:

- paragraph про `LAZY, BANDIT`

Добавить туда:

- лучший fixed-`K`
- comparison vs `WARM+LAZY+BANDIT`
- при необходимости `strict_feasible_rate`

### Если GNN шаг не выполнен

В статье можно оставить текущий downstream ablation, но:

- не усиливать claim,
- не писать так, будто `R1-C5` закрыт полностью classifier-level numbers.

---

## Как использовать результаты в response letter

### Если `grbtune` не выполнен

Для `R2-C4` писать:

- замечание не закрыто полностью;
- baseline acknowledged as limitation;
- direct tuning comparison left for future work.

### Если fixed-`K` не выполнен

Для `R1-C6` писать:

- формализация BANDIT добавлена;
- но количественный claim про marginal gain остается в основном textual.

### Если GNN P/R/F1 не выполнен

Для `R1-C5` писать:

- details и downstream ablation добавлены;
- classifier-level P/R/F1 остаются only partially addressed.

---

## Рекомендуемый план на 12 часов

### 0:00-0:15

- активировать `venv`
- проверить `PYTHONPATH`
- проверить `gurobipy`, `pandas`, `numpy`

### 0:15-2:30

- выполнить `grbtune` на `case118` и `case300`

### 2:30-5:30

- выполнить fixed-`K` ablation для `K=64`, `128`, `256`

### 5:30-6:00

- собрать `paper/tables/lazy_topk_ablation.csv`

### Остаток времени

- обновить paragraph в `paper.tex`
- при необходимости подправить wording в `response_to_reviewers.tex`
- опционально запустить `case89pegase` для `grbtune`

### GNN

- только если заранее починен mismatch между feature extractor и archived weights

---

## Итоговые рекомендации

- `grbtune` — обязательный reviewer-facing experiment
- fixed-`K` — лучший следующий шаг по отдаче на потраченное время
- GNN P/R/F1 — не обязательный в текущем состоянии репозитория
- все команды выше привязаны к текущему локальному layout проекта
- предупреждения в этом документе намеренно сформулированы жестко, чтобы не тратить compute budget на шаги, которые сейчас не готовы к запуску

---

## Короткая проверка после обновления статьи

Перед сборкой статьи проверьте:

- обновлен ли paragraph про `Solver tuning`
- обновлен ли paragraph про BANDIT / fixed-`K`
- не усилены ли claims по GNN без новых чисел
- не осталось ли в response letter формулировок, будто `R2-C4` или classifier-level часть `R1-C5` закрыты полностью без новых расчетов

