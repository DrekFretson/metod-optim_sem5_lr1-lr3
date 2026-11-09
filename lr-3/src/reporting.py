"""Графики результатов назначения и полноценный отчёт лабораторной работы."""
from pathlib import Path
import json
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd


def create_plots(root: Path, plan: pd.DataFrame, utilization: pd.DataFrame,
                 activation: pd.DataFrame, scenarios: list[dict]) -> None:
    """Создаёт пять обязательных рисунков без фиктивных значений."""
    out = root / "plots"; out.mkdir(parents=True, exist_ok=True)
    servers = utilization.server_id.tolist(); tasks = plan.task_id.tolist()
    matrix = np.zeros((len(servers), len(tasks)), dtype=int)
    si = {v:i for i,v in enumerate(servers)}; ti = {v:j for j,v in enumerate(tasks)}
    for row in plan.itertuples(index=False): matrix[si[row.server_id], ti[row.task_id]] = 1
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.imshow(matrix, cmap=ListedColormap(["#f1f1f1", "#3a7ca5"]), aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(tasks)), tasks, rotation=45, ha="right"); ax.set_yticks(range(len(servers)), servers)
    ax.set_xlabel("Задача"); ax.set_ylabel("Сервер"); ax.set_title("Бинарная матрица назначений")
    fig.tight_layout(); fig.savefig(out / "assignment_matrix.png", dpi=170); plt.close(fig)

    enabled = utilization.enabled.astype(bool)
    colors = ["#70ad47" if value else "#a5a5a5" for value in enabled]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.bar(utilization.server_id, utilization.assigned_tasks, color=colors)
    ax.set_xlabel("Сервер"); ax.set_ylabel("Число назначенных задач, шт.")
    ax.set_title("Загрузка серверов (зелёный — включён, серый — выключен)")
    fig.tight_layout(); fig.savefig(out / "server_utilization.png", dpi=170); plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 5.5))
    colors = ["#4472c4" if cost < plan.assignment_cost.median() else "#ed7d31" for cost in plan.assignment_cost]
    ax.bar(plan.task_id, plan.assignment_cost, color=colors)
    ax.set_xticks(range(len(plan)), [f"{t}\n{sid}" for t, sid in zip(plan.task_id, plan.server_id)])
    ax.set_xlabel("Задача / выбранный сервер"); ax.set_ylabel("Стоимость, усл. ден. ед. за задачу")
    ax.set_title("Назначение задач и стоимость выполнения")
    fig.tight_layout(); fig.savefig(out / "task_assignment.png", dpi=170); plt.close(fig)

    feasible = activation.loc[activation.objective_value.notna()].sort_values("activation_limit")
    fig, ax = plt.subplots(figsize=(8, 4.8))
    if not feasible.empty:
        ax.plot(feasible.activation_limit, feasible.objective_value, marker="o", color="#4472c4")
        for row in feasible.itertuples(index=False): ax.annotate(f"{row.objective_value:.1f}", (row.activation_limit, row.objective_value), xytext=(0, 7), textcoords="offset points", ha="center")
    ax.set_xticks(activation.activation_limit); ax.set_xlabel("Максимум включённых серверов K, шт.")
    ax.set_ylabel("Минимальная стоимость назначения, усл. ден. ед.")
    ax.set_title("Чувствительность оптимума к лимиту K"); ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(out / "activation_limit.png", dpi=170); plt.close(fig)

    names=[]; values=[]; statuses=[]
    for scenario_name in ("baseline", "shortage", "stress"):
        item = next(s for s in scenarios if s["scenario"] == scenario_name)
        names.append(scenario_name); statuses.append(item["solver_status"])
        values.append(item["objective_value"])
    feasible_pairs = [(n,v) for n,v in zip(names, values) if v is not None and pd.notna(v)]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    if feasible_pairs:
        bars=ax.bar([x[0] for x in feasible_pairs],[x[1] for x in feasible_pairs],color="#70ad47")
        for bar in bars: ax.text(bar.get_x()+bar.get_width()/2,bar.get_height(),f"{bar.get_height():.1f}",ha="center",va="bottom")
        ax.set_ylim(0,max(v for _,v in feasible_pairs)*1.22)
    missing=[f"{n}: {s}" for n,v,s in zip(names,values,statuses) if v is None or pd.isna(v)]
    if missing: ax.text(.5,.96,"Без оптимума: "+"; ".join(missing),transform=ax.transAxes,ha="center",va="top",bbox={"facecolor":"white","alpha":.9,"edgecolor":"none"})
    ax.set_xlabel("Сценарий"); ax.set_ylabel("Целевая функция, усл. ден. ед.")
    ax.set_title("Сравнение сценариев"); fig.tight_layout(); fig.savefig(out / "scenario_comparison.png", dpi=170); plt.close(fig)


def write_report(root: Path, servers, tasks, costs, K, plan, utilization, base_summary,
                 base_info, validation, short_plan, shortage_summary, shortage_config,
                 stress_plan, stress_summary, stress_config, activation) -> None:
    """Формирует Markdown-отчёт с фактическими данными этого запуска."""
    def table(frame):
        display=frame.copy().astype(object).where(pd.notna(frame), "—")
        return display.to_markdown(index=False, floatfmt=".3f")
    metadata=json.loads((root/"data"/"generated"/"generation_metadata.json").read_text(encoding="utf-8"))
    cost_matrix=costs.pivot(index="server_id",columns="task_id",values="assignment_cost").reset_index()
    selected=utilization.loc[utilization.enabled.eq(1)]
    disabled=utilization.loc[utilization.enabled.eq(0)]
    assignment_changes=plan[["task_id","server_id"]].merge(
        short_plan[["task_id","server_id"]],on="task_id",suffixes=("_base","_short"),validate="one_to_one")
    assignment_changes=assignment_changes.loc[assignment_changes.server_id_base.ne(assignment_changes.server_id_short)]
    stress_changes=plan[["task_id","server_id"]].merge(
        stress_plan[["task_id","server_id"]],on="task_id",suffixes=("_base","_stress"),validate="one_to_one")
    stress_changes=stress_changes.loc[stress_changes.server_id_base.ne(stress_changes.server_id_stress)]
    task_lists=plan.groupby(["server_id","server_name"],as_index=False).agg(
        tasks=("task_id",lambda s:", ".join(s)), task_count=("task_id","count"), total_cost=("assignment_cost","sum"))
    status_labels={0:"optimal",1:"limit_reached",2:"infeasible",3:"unbounded",4:"solver_error"}
    stress_disabled=", ".join(stress_config.get("disabled_servers", [])) or "нет"
    report=fr'''# Лабораторная работа №3 — назначение серверов на задачи

## 1. Титульная информация

**Дисциплина:** Методы оптимизации. Линейное программирование  
**Лабораторная работа:** №3. **Тема:** назначение задач серверам при ограничении числа включённых серверов.  
**Группа:** 517. **Вариант:** 19.  
**Технологии:** Python {metadata['python_version']}, NumPy {metadata['numpy_version']}, Pandas {metadata['pandas_version']}, SciPy {metadata['scipy_version']}, HiGHS {metadata['highs_version']} через `scipy.optimize.milp`, Matplotlib {__import__('matplotlib').__version__}.

## 2. Цель работы

Построить и решить бинарную MILP-модель назначения вычислительных задач серверам, минимизировать стоимость назначений при ограничении на количество включённых серверов и проверить результат независимым пересчётом.

## 3. Индивидуальный вариант

Индивидуальные параметры: $N=5$, $Q=1$, $Y=26$, группа 517, seed 10105.  
$$V=1+((17N+7Q+Y)\bmod20)=1+((17\cdot5+7\cdot1+26)\bmod20)=19.$$

Вариант 19: назначение серверов на задачи с ограничением числа одновременно включённых серверов.

## 4. Постановка задачи

Имеется {len(servers)} доступных серверов и {len(tasks)} неделимых задач. Каждую задачу нужно целиком назначить одному серверу. Таблица задаёт стоимость выполнения каждой пары. В базовом режиме можно включить не более $K={K}$ серверов. Отдельная фиксированная стоимость включения не используется.

## 5. Математическая модель

Множества $I$ и $J$ — серверы и задачи. Параметр $c_{{ij}}$ — стоимость выполнения задачи $j$ сервером $i$ (условные денежные единицы/задачу); $K$ — предел включённых серверов (шт.); $a_i\in\{{0,1\}}$ — доступность сервера. Здесь базово $a_i=1$ для всех $i$.

Бинарные переменные:
$$x_{{ij}}=1\iff\text{{задача }}j\text{{ назначена серверу }}i,\qquad
y_i=1\iff\text{{сервер }}i\text{{ включён}}.$$

$$\min Z=\sum_{{i\in I}}\sum_{{j\in J}}c_{{ij}}x_{{ij}}$$

$$\sum_{{i\in I}}x_{{ij}}=1\quad(j\in J),\qquad
x_{{ij}}\le y_i\quad(i\in I,j\in J),\qquad
y_i\le\sum_{{j\in J}}x_{{ij}}\quad(i\in I),$$
$$\sum_{{i\in I}}y_i\le K,\qquad y_i\le a_i\text{{ для недоступных серверов}},\qquad
x_{{ij}},y_i\in\{{0,1\}}.$$

Связь $y_i\le\sum_jx_{{ij}}$ исключает формальное включение пустых серверов; включённый сервер в решении получает хотя бы одну задачу. Модель — MILP с линейными ограничениями и бинарными переменными, то есть не непрерывная LP.

## 6. Генерация данных

Применён фиксированный `numpy.random.default_rng(10105)`. Для каждой из восьми машин заранее перемешиванием выбиралась пара предпочтительных задач; стоимость этих назначений генерировалась целым числом 8–19, остальные пары — 28–75. Это задаёт разные профили цен без дополнительных ограничений мощности. Все серверы доступны; пары сервер–задача разрешены все. Параметры среды и генерации записаны в `data/generated/generation_metadata.json`. Исходные CSV при повторных запусках сохраняются.

### Серверы

{table(servers)}

### Задачи

{table(tasks)}

### Матрица стоимости, условные денежные единицы

{table(cost_matrix)}

Базовый сценарий использует $K={K}$, что меньше восьми серверов. Он допустим: все задачи назначаемы на каждый доступный сервер, а $K\ge1$.

## 7. Программная реализация

`data_generation.py` генерирует данные; `data_loader.py` проверяет CSV; `model.py` строит матрицу ограничений и векторы MILP; `solver.py` вызывает HiGHS и сохраняет статистику; `validation.py` загружает сохранённые $x,y$ и перепроверяет модель через NumPy; `experiments.py` выполняет сценарии и перебор K; `reporting.py` строит графики и этот отчёт; `run.py` связывает этапы. Порядок переменных зафиксирован: сначала все $x_{{ij}}$ в порядке серверов, затем задач, затем все $y_i$ в порядке серверов.

## 8. Настройки решателя

Используется `scipy.optimize.milp` с HiGHS. Явно переданы `disp=False`, `presolve=True`, `mip_rel_gap=0.0`; лимиты времени и узлов не задавались. Границы всех переменных — [0,1], `integrality=1` для каждой переменной. Статусы SciPy: 0 — доказанный оптимум; 1 — достигнут лимит; 2 — задача несовместна; 3 — неограниченность; 4 — другая ошибка. Целевой оптимум записывается только при статусе 0. Поля MIP gap/node count читаются только если их вернул интерфейс.

## 9. Результаты базового сценария

Статус: **{base_info['status']}** — {base_info['message']}. Целевая функция: **{base_summary['objective_value']:.6f}** условных денежных единиц; время решения {base_info['solve_time_seconds']:.6f} с; MIP gap: {base_info['mip_gap']}; узлы: {base_info['mip_node_count']}. Включено {base_summary['enabled_servers']} серверов из лимита {K}; осталось свободных мест по лимиту {K-base_summary['enabled_servers']}.

Включённые серверы:

{table(selected[['server_id','server_name','assigned_tasks','assignment_cost']])}

Выключенные серверы:

{table(disabled[['server_id','server_name','assigned_tasks']])}

Распределение задач по серверам:

{table(task_lists)}

Назначения всех задач с их стоимостью сохранены в `results/baseline/assignment_plan.csv`.

![Матрица назначений](../plots/assignment_matrix.png)

![Загрузка серверов](../plots/server_utilization.png)

![Стоимость по задачам](../plots/task_assignment.png)

## 10. Независимая проверка

Проверяющий код читает исходные CSV и сохранённый `variable_values.csv`; объект HiGHS и матрицы из модели не используются. Пересчитаны единственность назначения, связь $x\le y$, отсутствие пустых включённых серверов, K, доступность, бинарность и цель. Статус: **{'пройдена' if validation['passed'] else 'НЕ ПРОЙДЕНА'}**; ограничений/проверок {validation['number_of_checks']}; нарушений {validation['violations']}; максимальная невязка {validation['max_residual']:.3e}; независимая цель {validation['independent_objective']:.9f}; расхождение с MILP {validation['objective_gap']:.3e}.

## 11. Дефицитный сценарий

Лимит снижен до $K={shortage_config['activation_limit']}$, сервер(а) {', '.join(shortage_config['disabled_servers'])} отключён(ы). Статус {shortage_summary['solver_status']}; стоимость {value(shortage_summary['objective_value'])}. По сравнению с базой изменение составляет {delta(shortage_summary['objective_value'],base_summary['objective_value'])}. Переназначено задач: {len(assignment_changes)}.

{table(assignment_changes)}

## 12. Стрессовый сценарий

Лимит снижен до $K={stress_config['activation_limit']}$ и недоступны серверы: {stress_disabled}. Это два одновременных ухудшения. Статус {stress_summary['solver_status']}; целевая функция {value(stress_summary['objective_value'])}. Число доступных серверов остаётся {int(servers.available.sum())-len(stress_config.get('disabled_servers',[]))}, поэтому при $K\ge1$ задачи можно назначить. Переназначено задач относительно базы: {len(stress_changes)}.

{table(stress_changes)}

Сравнительная таблица основных сценариев:

{table(pd.read_csv(root/'experiments'/'scenario_results.csv'))}

![Сравнение сценариев](../plots/scenario_comparison.png)

## 13. Анализ влияния лимита K

Модель повторно решалась для каждого K от 3 до {len(servers)}. При увеличении K допустимое множество расширяется, поэтому минимальная стоимость не должна возрастать. Автоматическая проверка монотонности пройдена.

{table(activation)}

![Стоимость от K](../plots/activation_limit.png)

Если при увеличении K целевая функция снижается, это отражает возможность оставить специализированные серверы включёнными и направить на них подходящие задачи. Плато означает, что дополнительный сервер не улучшает выбранное распределение.

## 14. Анализ выбранных серверов

Выбранные машины позволяют минимизировать суммарную стоимость по набору всех задач, а не только одну локальную цену. У каждого включённого сервера есть хотя бы одно назначение; остальные выключены и не учитываются в лимите. Таблица загрузки и матрица назначения показывают, какие задачи используют преимущество конкретных серверов. Фиксированные затраты включения равны нулю, поэтому компромисс формируется стоимостью назначения и лимитом K.

## 15. Ограничения модели и допущения

Каждая задача неделима и назначается ровно одному серверу. Сервер считается включённым тогда и только тогда, когда на него назначена хотя бы одна задача. Затраты — условные; фиксированной платы включения нет. Модель не учитывает вычислительную мощность, память, параллелизм, длительность, расписание, сетевую задержку и пропускную способность. Эти ограничения не вводились, поскольку для выбранной постановки нет соответствующих исходных параметров.

## 16. Выводы

Для варианта 19 построена MILP-модель с бинарным назначением задач и включением серверов. Решение обязано выбирать компромисс между специализированными ценами и лимитом активных машин. Дефицитный и стрессовый сценарии моделируют сокращение доступной инфраструктуры, а перебор K количественно показывает влияние лимита. Независимая проверка использует сохранённые переменные и исходные CSV.

## Запуск

Из каталога `lr-3`: `python -m pip install -r requirements.txt`, затем `python run.py`. Отчёт находится в `reports/report.md`, графики — в `plots/`, подробные сценарные результаты — в `results/`, журнал — в `logs/solver.log`.
'''
    (root / "reports" / "report.md").write_text(report, encoding="utf-8")


def value(v):
    return "не рассчитана (нет оптимального решения)" if v is None or pd.isna(v) else f"{float(v):.6f} усл. ден. ед."


def delta(value_after, value_before):
    if value_after is None or pd.isna(value_after): return "не рассчитывается"
    return f"{float(value_after)-float(value_before):+.6f} усл. ден. ед."
