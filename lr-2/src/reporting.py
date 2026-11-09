"""Построение пяти графиков и итогового русскоязычного отчёта."""
from pathlib import Path
import json
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


def create_plots(root: Path, plan: pd.DataFrame, warehouses: pd.DataFrame,
                 stores: pd.DataFrame, cases: dict, route_analysis: pd.DataFrame) -> None:
    """Строит план, затраты, доступность дуг, запасы и сценарии."""
    out = root / "plots"; out.mkdir(parents=True, exist_ok=True)
    wids = warehouses.warehouse_id.tolist(); sids = stores.store_id.tolist()
    matrix = plan.pivot(index="warehouse_id", columns="store_id", values="quantity").reindex(index=wids, columns=sids).fillna(0)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    image = ax.imshow(matrix.to_numpy(), cmap="Blues", aspect="auto")
    ax.set_xticks(range(len(sids)), sids); ax.set_yticks(range(len(wids)), wids)
    ax.set_xlabel("Магазин"); ax.set_ylabel("Склад"); ax.set_title("Оптимальный план перевозок, ед. продукции")
    for i in range(len(wids)):
        for j in range(len(sids)):
            if matrix.iloc[i, j] > 1e-8: ax.text(j, i, f"{matrix.iloc[i,j]:.0f}", ha="center", va="center", color="black")
    fig.colorbar(image, ax=ax, label="Количество продукции, ед."); fig.tight_layout(); fig.savefig(out / "transportation_plan.png", dpi=170); plt.close(fig)

    used_costs = plan.loc[plan.quantity.fillna(0).gt(1e-8)].sort_values("total_cost", ascending=False)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    if not used_costs.empty:
        labels = [f"{r.warehouse_id}→{r.store_id}" for r in used_costs.itertuples()]
        ax.bar(labels, used_costs.total_cost, color="#ed7d31")
        ax.tick_params(axis="x", labelrotation=35)
    ax.set_xlabel("Используемый маршрут"); ax.set_ylabel("Затраты, усл. ден. ед.")
    ax.set_title("Вклад маршрутов в транспортные затраты"); fig.tight_layout(); fig.savefig(out / "transportation_costs.png", dpi=170); plt.close(fig)

    route_matrix = np.zeros((len(wids), len(sids)), dtype=int)
    wi = {v:i for i,v in enumerate(wids)}; si = {v:j for j,v in enumerate(sids)}
    for r in route_analysis.itertuples(index=False):
        route_matrix[wi[r.warehouse_id], si[r.store_id]] = 0 if not r.allowed else (2 if r.used else 1)
    cmap = ListedColormap(["#d9534f", "#eeeeee", "#2e8b57"]); norm = BoundaryNorm([-0.5,0.5,1.5,2.5], cmap.N)
    fig, ax = plt.subplots(figsize=(8, 4.8)); ax.imshow(route_matrix, cmap=cmap, norm=norm, aspect="auto")
    ax.set_xticks(range(len(sids)), sids); ax.set_yticks(range(len(wids)), wids)
    ax.set_xlabel("Магазин"); ax.set_ylabel("Склад"); ax.set_title("Доступность и использование маршрутов")
    ax.legend(handles=[Patch(color="#d9534f", label="Запрещён"), Patch(color="#eeeeee", label="Разрешён, не используется"), Patch(color="#2e8b57", label="Используется")], loc="upper center", bbox_to_anchor=(0.5,-0.17), ncol=3)
    fig.tight_layout(); fig.savefig(out / "route_availability.png", dpi=170, bbox_inches="tight"); plt.close(fig)

    usage = warehouses.merge(plan.groupby("warehouse_id", as_index=False).quantity.sum(min_count=1), on="warehouse_id", how="left").fillna({"quantity":0})
    usage["unused"] = usage.supply - usage.quantity
    fig, ax = plt.subplots(figsize=(8, 4.8)); x = np.arange(len(usage))
    ax.bar(x, usage.quantity, label="Отгружено", color="#4472c4")
    ax.bar(x, usage.unused, bottom=usage.quantity, label="Остаток", color="#a5a5a5")
    ax.set_xticks(x, usage.warehouse_id); ax.set_xlabel("Склад"); ax.set_ylabel("Продукция, ед.")
    ax.set_title("Использование складских запасов"); ax.legend(); fig.tight_layout(); fig.savefig(out / "warehouse_utilization.png", dpi=170); plt.close(fig)

    labels=[]; values=[]; infeasible=[]
    for name, label in (("baseline","Базовый"),("shortage","Дефицитный"),("stress","Стрессовый")):
        c=cases[name]; labels.append(label)
        if c["objective_value"] is None: infeasible.append(f"{label}: {c['status']}")
        else: values.append((label,c["objective_value"]))
    fig, ax = plt.subplots(figsize=(8, 4.8))
    if values:
        bars=ax.bar([v[0] for v in values],[v[1] for v in values],color="#70ad47")
        for b in bars: ax.text(b.get_x()+b.get_width()/2,b.get_height(),f"{b.get_height():.1f}",ha="center",va="bottom")
    ax.set_xlabel("Сценарий"); ax.set_ylabel("Общие затраты, усл. ден. ед."); ax.set_title("Сравнение транспортных сценариев")
    if values: ax.set_ylim(0, max(v[1] for v in values) * 1.25)
    if infeasible: ax.text(.5,.96,"Нет допустимого решения: "+"; ".join(infeasible),transform=ax.transAxes,ha="center",va="top",bbox={"facecolor":"white","alpha":0.85,"edgecolor":"none"})
    fig.tight_layout(); fig.savefig(out / "scenario_comparison.png", dpi=170); plt.close(fig)


def write_report(root: Path, warehouses, stores, costs, routes, plan, shortage_plan, warehouse_use,
                 store_service, validation, baseline_info, shortage_config, stress_config,
                 cases, shadows, shadow_checks, route_analysis, unrestricted) -> None:
    """Записывает отчёт с таблицами только из фактически вычисленных результатов."""
    def table(df):
        display = df.copy().astype(object).where(pd.notna(df), "—")
        return display.to_markdown(index=False, floatfmt=".3f")
    metadata = json.loads((root / "data" / "generated" / "generation_metadata.json").read_text(encoding="utf-8"))
    allowed = int(routes.allowed.sum()); total_routes = len(routes); forbidden = total_routes - allowed
    used = route_analysis.loc[route_analysis.used]
    wh_options = routes.groupby("warehouse_id").allowed.sum().rename("allowed_routes").reset_index()
    st_options = routes.groupby("store_id").allowed.sum().rename("allowed_routes").reset_index()
    costly = used.sort_values("total_cost", ascending=False).head(5)
    flow_diff = plan[["warehouse_id","store_id","quantity"]].merge(
        shortage_plan[["warehouse_id","store_id","quantity"]],
        on=["warehouse_id","store_id"], suffixes=("_baseline","_shortage"), validate="one_to_one")
    flow_diff["change"] = flow_diff.quantity_shortage - flow_diff.quantity_baseline
    flow_diff = flow_diff.loc[flow_diff.change.abs() > 1e-7].sort_values("change", key=lambda s: s.abs(), ascending=False)
    stress_short = sum(float(v["before"]-v["after"]) for v in stress_config["warehouse_supply_reduction"].values())
    stress_add = sum(float(v["after"]-v["before"]) for v in stress_config["demand_increase"].values())
    stress_supply = float(warehouses.supply.sum()) - stress_short
    stress_demand = float(stores.demand.sum()) + stress_add
    route_change = stress_config.get("additional_forbidden_route")
    if route_change:
        affected_store = route_change["store_id"]
        before_links = int(routes.loc[routes.store_id.eq(affected_store), "allowed"].sum())
        after_links = before_links - 1
        route_impact = f"Магазин {affected_store} теряет связь со складом {route_change['warehouse_id']}: доступных входов было {before_links}, осталось {after_links}; полной изоляции нет."
    else:
        route_impact = "Дополнительный маршрут не пришлось закрывать: подходящего использованного маршрута с альтернативой не было."
    least_connected = st_options.sort_values("allowed_routes").iloc[0]
    scarce = shadows.loc[(shadows.constraint_type == "supply") & (shadows.marginal < -1e-8)].sort_values("marginal")
    most_valuable = scarce.iloc[0] if not scarce.empty else None
    cheapest_route = route_analysis.loc[route_analysis.allowed.eq(1)].sort_values("unit_cost").iloc[0]
    if most_valuable is not None:
        supply_value_text = f"Склад {most_valuable.entity_id} имеет наиболее отрицательный marginal {most_valuable.marginal:.3f}; дополнительная единица запаса снижает минимум примерно на {abs(most_valuable.marginal):.3f} в локальном диапазоне."
    else:
        supply_value_text = "В базовом оптимуме нет склада с ненулевой ценой запаса."
    shortage = cases["shortage"]; stress = cases["stress"]; base = cases["baseline"]
    report = fr'''# Лабораторная работа №2 — транспортная задача

## 1. Титульная информация

**Дисциплина:** Методы оптимизации. Линейное программирование  
**Тема:** оптимальное распределение продукции при запрете части маршрутов.  
**Группа:** 517; **вариант:** 19.  
**Технологии фактического запуска:** Python {metadata['python_version']}, NumPy {metadata['numpy_version']}, Pandas {metadata['pandas_version']}, SciPy {metadata['scipy_version']}, HiGHS {metadata['highs_version']} через `scipy.optimize.linprog`, Matplotlib {__import__('matplotlib').__version__}.

## 2. Цель работы

Построить транспортную LP-модель с явным исключением запрещённых маршрутов, минимизировать стоимость доставки, проверить решение независимым расчётом и изучить влияние ограничений и сценарных изменений.

## 3. Индивидуальный вариант

Использованы $N=5$, $Q=1$, $Y=26$, группа 517, seed 10105.  
$$V=1+((17N+7Q+Y)\bmod 20)=1+((17\cdot5+7\cdot1+26)\bmod20)=19.$$

Вариант 19: транспортная задача с запретом части маршрутов.

## 4. Постановка задачи

Пять складов отправляют продукцию шести магазинам. Для каждого склада известен запас в единицах продукции, для каждого магазина — точный спрос. У каждой пары «склад–магазин» есть положительная стоимость доставки одной единицы, но часть дуг закрыта. Запрещённые пары не получают переменных в LP и в сохранённом плане имеют нулевой поток.

## 5. Математическая модель

Пусть $I$ — склады, $J$ — магазины, $A\subseteq I\times J$ — разрешённые маршруты. Параметры: $s_i$ — запас (ед. продукции), $d_j$ — спрос (ед. продукции), $c_{{ij}}$ — стоимость единицы по маршруту (условные денежные единицы/ед.). Переменная $x_{{ij}}\ge0$ — объём поставки по разрешённому маршруту (ед.).

$$\min Z=\sum_{{(i,j)\in A}}c_{{ij}}x_{{ij}}$$

$$\sum_{{j:(i,j)\in A}}x_{{ij}}\le s_i\quad(i\in I),\qquad
\sum_{{i:(i,j)\in A}}x_{{ij}}=d_j\quad(j\in J),\qquad x_{{ij}}\ge0.$$

Переменные создаются только для $A$. Это агрегированная потоковая модель регулярных поставок; непрерывные значения представляют средний объём за период. Для единичных неделимых отправок требовалась бы целочисленная постановка. Запрет не моделируется большой стоимостью: конечный штраф может оказаться выгоднее других маршрутов и потому не гарантирует нулевой поток.

## 6. Генерация данных

Использован `numpy.random.default_rng(10105)`. Спрос магазинов равномерно генерировался целыми числами 38–60. Разрешённость маршрутов выбиралась с вероятностью 0.50, затем добавлялся связный backbone $W_i\to S_i,S_{{i+1}}$; при необходимости каждому магазину обеспечивалось не менее двух входящих дуг. Стоимости генерировались в диапазоне 7–30 с дополнительной индексной составляющей расстояния. Начальные запасы равномерно заданы как $\lceil\sum_jd_j/|I|\rceil+4$; при необходимости мощности увеличивались, пока сеть не становилась допустимой. Перед сохранением отдельная LP-проверка HiGHS удостоверяет выполнимость по реальным разрешённым дугам. Исходные CSV при повторных запусках не меняются.

### Склады

{table(warehouses)}

### Магазины

{table(stores)}

### Примеры транспортных затрат (первые 12 пар)

{table(costs.head(12))}

## 7. Программная реализация

`data_generation.py` создаёт данные и проверяет выполнимость; `data_loader.py` проверяет таблицы и ключи; `model.py` строит матрицы только по разрешённым маршрутам; `solver.py` вызывает HiGHS; `validation.py` независимо пересчитывает сохранённый план; `sensitivity.py` разбирает маргиналы и проверяет их возмущениями RHS; `experiments.py` выполняет сценарии и анализ дуг; `reporting.py` формирует графики и этот отчёт; `run.py` управляет запуском. Для $n_A$ разрешённых маршрутов модель содержит ровно $n_A$ переменных.

## 8. Настройки решателя

Решатель — HiGHS из SciPy, `scipy.optimize.linprog(method="highs")`; переменные ограничены снизу нулём, для запасов заданы `A_ub x <= b_ub`, спрос представлен `A_eq x = b_eq`. Явные настройки tolerances не переопределялись: используются значения SciPy/HiGHS по умолчанию. Статусы SciPy: 0 — оптимум; 1 — лимит; 2 — несовместность; 3 — неограниченность; 4 — ошибка решателя. Численные данные среды записаны в `data/generated/generation_metadata.json`.

## 9. Результаты базового сценария

Статус: **{base['status']}**; оптимальные транспортные затраты: **{base['objective_value']:.6f}** условных денежных единиц; время решения {base['solve_time_seconds']:.6f} с; итерации: {base['iterations']}. Отправлено {base['used_supply']:.3f} из {base['total_supply']:.3f} единиц, совокупный спрос — {base['total_demand']:.3f} единиц. Использовано {base['number_of_used_routes']} из {allowed} разрешённых маршрутов.

### План всех потенциальных маршрутов

{table(plan[['warehouse_id','store_id','allowed','unit_cost','quantity','total_cost']])}

### Использование складов

{table(warehouse_use[['warehouse_id','supply','shipped','unused']])}

### Обслуживание магазинов

{table(store_service[['store_id','demand','received','difference']])}

Самые затратные использованные маршруты:

{table(costly[['warehouse_id','store_id','quantity','unit_cost','total_cost','share_of_total_transport_cost']])}

![План перевозок](../plots/transportation_plan.png)

![Вклад маршрутов в затраты](../plots/transportation_costs.png)

## 10. Независимая проверка

Проверяющий модуль повторно загружает сохранённый `transportation_plan.csv` и CSV исходных данных. Независимо пересчитаны отгрузки каждого склада, поступления каждому магазину, остатки, запрещённые дуги, неотрицательность и цель. Допуск $10^{{-7}}$. Проверка: **{'пройдена' if validation['passed'] else 'НЕ ПРОЙДЕНА'}**; максимальная невязка {validation['max_residual']:.3e}; пересчитанная цель {validation['independent_objective']:.9f}; расхождение с HiGHS {validation['objective_gap']:.3e}.

{table(pd.read_csv(root/'results'/'baseline'/'validation.csv'))}

## 11. Анализ запрещённых маршрутов

Всего пар: **{total_routes}**, разрешено **{allowed}**, запрещено **{forbidden}** ({100*forbidden/total_routes:.1f}%). Запреты являются структурными: поток по ним не входит в вектор решения. Разрешённость и использование отмечены на карте.

Число доступных связей по складам:

{table(wh_options)}

Число доступных связей по магазинам:

{table(st_options)}

Без запретов стоимость равна {unrestricted['objective_value']:.6f}; с запретами — {base['objective_value']:.6f}. Разница стоимости от ограничения дуг: {base['objective_value']-unrestricted['objective_value']:+.6f} ({(base['objective_value']/unrestricted['objective_value']-1)*100:+.2f}%). Это сравнение сохраняется отдельно от трёх основных сценариев.

![Доступность маршрутов](../plots/route_availability.png)

## 12. Сценарные эксперименты

Сводные статусы, цели, запасы, спрос, использованный ресурс, число активных маршрутов и время приведены в таблице ниже.

{table(pd.read_csv(root/'experiments'/'scenario_results.csv'))}

**Дефицитный сценарий.** Сокращён запас склада {shortage_config['warehouse_id']} на {100*shortage_config['reduction']:.0f}% ({shortage_config['supply_before']:.3f} → {shortage_config['supply_after']:.3f}). Статус: {shortage['status']}; изменение стоимости относительно базы: {fmt_delta(shortage['objective_value'],base['objective_value'])}. Потоки по изменившимся связям:

{table(flow_diff[['warehouse_id','store_id','quantity_baseline','quantity_shortage','change']])}

В JSON сохранена точно применённая корректировка.

**Стрессовый сценарий.** Одновременно запас двух активно используемых складов сокращён на 30%, спрос двух магазинов увеличен на 20%, а один ранее разрешённый использованный маршрут запрещён. Потеря запаса {stress_short:.3f} единиц, прирост спроса {stress_add:.3f} единиц. Статус: **{stress['status']}**; целевая функция не приводится при отсутствии оптимального решения. Совокупный запас {stress_supply:.3f} меньше спроса {stress_demand:.3f} на {stress_demand-stress_supply:.3f} единиц, что само по себе объясняет infeasible. {route_impact}

![Сценарии](../plots/scenario_comparison.png)

## 13. Анализ теневых цен

HiGHS возвращает маргиналы RHS. Для склада передано $A_{{ub}}x\le s$, поэтому marginal обычно неположителен: увеличение запаса меняет минимум приблизительно на `marginal × Δs`. Для спроса задано равенство, и маргинал показывает изменение минимума при увеличении точного спроса. Неактивные ограничения с нулевым маргиналом локальной ценности не имеют.

{table(shadows)}

Проверки малых изменений RHS (+1 единица) для ненулевых маргиналов:

{table(shadow_checks)}

Теневые цены локальны: совпадение прогноза с фактом относится только к малому изменению, пока не меняется оптимальная структура потоков.

## 14. Анализ результатов

{supply_value_text} Наименьший выбор поставщиков у магазина {least_connected['store_id']}: {int(least_connected['allowed_routes'])} входящих разрешённых маршрута. Самый дешёвый разрешённый путь — {cheapest_route.warehouse_id}→{cheapest_route.store_id} по {cheapest_route.unit_cost:.3f} за единицу; он {'используется' if cheapest_route.used else 'не используется в оптимуме'}. Наиболее затратные используемые дуги показаны в таблице плана. Запреты повышают минимум на {base['objective_value']-unrestricted['objective_value']:.3f} относительно сети, где все дуги открыты. Дефицитный сценарий сокращает запас {shortage_config['warehouse_id']} и повышает цель на {shortage['objective_value']-base['objective_value']:.3f}; стрессовый снижает суммарную мощность ниже спроса.

## 15. Ограничения модели и допущения

Переменные непрерывны, поскольку это агрегированный поток продукции за период. Не моделируются фиксированные затраты открытия маршрута, пропускные способности отдельных дуг, временные окна, сроки доставки, потери, разные классы товара, транзитные узлы и случайность тарифов. Запасы — верхние границы, спрос удовлетворяется точно. Стоимости условные; физическая единица товара считается однородной.

## 16. Выводы

Для варианта 19 построена транспортная LP-модель с пятью складами, шестью магазинами и явным исключением запрещённых дуг из множества переменных. Базовое распределение найдено HiGHS и независимо перепроверено по сохранённым CSV. Отдельные сценарии показывают последствия сокращения складских ресурсов и одновременного ухудшения спроса/сети; unrestricted-сравнение оценивает стоимость запретных связей. Теневые оценки связывают локальное изменение запасов и спроса с изменением транспортных расходов.

## Запуск

Из корня проекта: `python -m pip install -r requirements.txt`; затем `cd lr-2 && python run.py`. Данные: `data/generated/`; сценарии: `data/scenarios/`; результаты: `results/`; сводки: `experiments/`; графики: `plots/`; журнал: `logs/solver.log`.
'''
    (root / "reports" / "report.md").write_text(report, encoding="utf-8")


def fmt_delta(value, base):
    return "не рассчитывается (нет оптимального решения)" if value is None else f"{value-base:+.6f} условных денежных единиц"
