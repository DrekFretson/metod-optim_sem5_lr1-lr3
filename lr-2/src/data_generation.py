"""Генерация сбалансированных и заведомо допустимых транспортных данных."""
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy.optimize import linprog

SEED = 10105
N_WAREHOUSES = 5
N_STORES = 6


def _highs_version() -> str:
    """Возвращает версию HiGHS из адаптера SciPy, если API её экспортирует."""
    try:
        from scipy.optimize._highspy import _core
        return ".".join(str(getattr(_core, key)) for key in
                         ("HIGHS_VERSION_MAJOR", "HIGHS_VERSION_MINOR", "HIGHS_VERSION_PATCH"))
    except (ImportError, AttributeError):
        return "bundled with SciPy; version not exposed by public API"


def _ensure_feasible(routes: np.ndarray, supply: np.ndarray, demand: np.ndarray) -> bool:
    """Проверяет выполнимость сети потока с запрещёнными дугами."""
    n_w, n_s = routes.shape
    edges = [(i, j) for i in range(n_w) for j in range(n_s) if routes[i, j]]
    if any(not routes[:, j].any() for j in range(n_s)) or supply.sum() < demand.sum():
        return False
    a_eq = np.zeros((n_s, len(edges)))
    a_ub = np.zeros((n_w, len(edges)))
    for k, (i, j) in enumerate(edges):
        a_eq[j, k] = 1
        a_ub[i, k] = 1
    result = linprog(np.zeros(len(edges)), A_eq=a_eq, b_eq=demand,
                     A_ub=a_ub, b_ub=supply, bounds=(0, None), method="highs")
    return bool(result.success)


def generate_data(directory: Path) -> None:
    """Создаёт исходные CSV один раз, сохраняя их при последующих запусках."""
    directory.mkdir(parents=True, exist_ok=True)
    names = ("warehouses.csv", "stores.csv", "transportation_costs.csv", "routes.csv", "generation_metadata.json")
    if all((directory / name).is_file() for name in names):
        return

    rng = np.random.default_rng(SEED)
    warehouse_ids = [f"W{i+1}" for i in range(N_WAREHOUSES)]
    store_ids = [f"S{j+1}" for j in range(N_STORES)]
    demand = rng.integers(38, 61, size=N_STORES)

    # Backbone W1→S1,S2; ...; W5→S5,S6 гарантирует связность спроса.
    allowed = rng.random((N_WAREHOUSES, N_STORES)) < 0.50
    for i in range(N_WAREHOUSES):
        allowed[i, i] = True
        allowed[i, i + 1] = True
    # Сохраняем каждый склад в сети и минимум два варианта для магазинов,
    # когда возможно, затем проверяем полную транспортную выполнимость.
    for j in range(N_STORES):
        if allowed[:, j].sum() < 2:
            candidates = np.argsort(rng.random(N_WAREHOUSES))
            allowed[candidates[:2], j] = True
    unit_cost = rng.integers(7, 31, size=(N_WAREHOUSES, N_STORES)).astype(float)
    distance_component = np.fromfunction(lambda i, j: 2.5 * np.abs(i - (j * (N_WAREHOUSES - 1) / (N_STORES - 1))), unit_cost.shape)
    unit_cost += distance_component

    # Запасы делятся между складами почти равномерно, чтобы отдельные дешёвые
    # источники могли иметь реальную ценность; выполнимость подтверждается LP.
    # Конструктивный cheapest-arc план остаётся запасным допустимым решением.
    supply = np.full(N_WAREHOUSES, int(np.ceil(demand.sum() / N_WAREHOUSES)) + 4, dtype=int)

    # Защитная корректировка на случай будущих изменений генератора/структуры.
    attempts = 0
    while not _ensure_feasible(allowed, supply, demand):
        supply += 1
        attempts += 1
        if attempts > int(demand.sum()):
            raise RuntimeError("Не удалось обеспечить допустимость транспортной сети")

    warehouses = pd.DataFrame({
        "warehouse_id": warehouse_ids,
        "warehouse_name": [f"Склад {i+1}" for i in range(N_WAREHOUSES)],
        "supply": supply.astype(int),
    })
    stores = pd.DataFrame({
        "store_id": store_ids,
        "store_name": [f"Магазин {j+1}" for j in range(N_STORES)],
        "demand": demand.astype(int),
    })
    costs, route_rows = [], []
    for i, wid in enumerate(warehouse_ids):
        for j, sid in enumerate(store_ids):
            costs.append({"warehouse_id": wid, "store_id": sid, "unit_cost": float(unit_cost[i, j])})
            route_rows.append({"warehouse_id": wid, "store_id": sid, "allowed": int(allowed[i, j])})
    pd.DataFrame(costs).to_csv(directory / "transportation_costs.csv", index=False)
    pd.DataFrame(route_rows).to_csv(directory / "routes.csv", index=False)
    warehouses.to_csv(directory / "warehouses.csv", index=False)
    stores.to_csv(directory / "stores.csv", index=False)

    metadata = {
        "N": 5, "Q": 1, "Y": 26, "group": 517,
        "variant_formula": "1 + ((17*N + 7*Q + Y) mod 20)",
        "variant": 1 + ((17 * 5 + 7 * 1 + 26) % 20), "seed": SEED,
        "python_version": platform.python_version(), "numpy_version": np.__version__,
        "pandas_version": pd.__version__, "scipy_version": scipy.__version__,
        "highs_version": _highs_version(),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "warehouses": N_WAREHOUSES, "stores": N_STORES,
        "potential_routes": N_WAREHOUSES * N_STORES,
        "allowed_routes": int(allowed.sum()), "forbidden_routes": int((~allowed).sum()),
        "algorithm": "default_rng(10105); random integer demands 38..60; random route mask p=0.50 with chain backbone and at least two incoming arcs per store; route costs random 7..30 plus an index distance component; initial capacities are balanced as ceil(total demand / warehouse count)+4, expanded together if needed; feasibility is independently checked with linprog/HiGHS.",
        "ranges": {"store_demand_units": [38, 60], "unit_cost_per_unit": "7..30 plus index distance component"},
        "units": {"supply": "product units", "demand": "product units", "unit_cost": "currency units per product unit", "total_cost": "currency units"},
        "assumptions": ["Continuous flow quantities represent aggregated shipments.", "Every store demand is met exactly.", "Warehouse supply is an upper bound and unused supply is allowed.", "Forbidden routes are excluded from decision variables.", "No route capacity, fixed activation cost, time window, or loss is modeled."],
        "feasibility_check": "Passed by a separate zero-cost transportation LP before saving data.",
    }
    (directory / "generation_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
