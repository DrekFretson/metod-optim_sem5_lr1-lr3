"""Загрузка таблиц ЛР2 и контроль структуры транспортных данных."""
from pathlib import Path
import pandas as pd


def load_data(directory: Path):
    """Читает четыре CSV, проверяет ключи, значения и наличие разрешённых связей."""
    try:
        warehouses = pd.read_csv(directory / "warehouses.csv")
        stores = pd.read_csv(directory / "stores.csv")
        costs = pd.read_csv(directory / "transportation_costs.csv")
        routes = pd.read_csv(directory / "routes.csv")
    except (OSError, ValueError) as exc:
        raise ValueError(f"Ошибка чтения исходных CSV: {exc}") from exc
    required = [
        (warehouses, {"warehouse_id", "warehouse_name", "supply"}),
        (stores, {"store_id", "store_name", "demand"}),
        (costs, {"warehouse_id", "store_id", "unit_cost"}),
        (routes, {"warehouse_id", "store_id", "allowed"}),
    ]
    for frame, columns in required:
        if not columns.issubset(frame.columns):
            raise ValueError(f"Не хватает столбцов {columns - set(frame.columns)}")
        keys = (["warehouse_id", "store_id"] if {"warehouse_id", "store_id"}.issubset(columns)
                else [key for key in ("warehouse_id", "store_id") if key in columns])
        if frame.duplicated(keys).any():
            raise ValueError("Обнаружены повторяющиеся пары склад–магазин")
        id_columns = [key for key in ("warehouse_id", "store_id", "warehouse_name", "store_name") if key in columns]
        if frame[id_columns].isna().any().any():
            raise ValueError("Идентификаторы и названия не должны быть пустыми")
    if len(warehouses) < 4 or len(stores) < 6:
        raise ValueError("Нужно не менее четырёх складов и шести магазинов")
    for frame, column in ((warehouses, "supply"), (stores, "demand")):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
        if not np_is_integer(frame[column]) or (frame[column] <= 0).any():
            raise ValueError(f"Поле {column} должно содержать целые числа")
    costs["unit_cost"] = pd.to_numeric(costs.unit_cost, errors="coerce")
    if not costs.unit_cost.map(lambda x: __import__("math").isfinite(float(x))).all() or (costs.unit_cost <= 0).any():
        raise ValueError("Транспортные затраты должны быть положительными конечными числами")
    if not routes.allowed.isin([0, 1]).all():
        raise ValueError("routes.allowed может содержать только 0 или 1")
    expected = len(warehouses) * len(stores)
    if len(costs) != expected or len(routes) != expected:
        raise ValueError("Для каждой пары склад–магазин должны быть заданы стоимость и статус маршрута")
    valid_w = set(warehouses.warehouse_id); valid_s = set(stores.store_id)
    for frame in (costs, routes):
        if set(frame.warehouse_id) != valid_w or set(frame.store_id) != valid_s:
            raise ValueError("В CSV маршрутов встречаются неизвестные склады или магазины")
    merged = routes.merge(costs, on=["warehouse_id", "store_id"], validate="one_to_one")
    if not (merged.allowed == 0).any() or not (merged.allowed == 1).any():
        raise ValueError("Должны присутствовать разрешённые и запрещённые маршруты")
    if (merged.groupby("store_id").allowed.sum() < 1).any():
        raise ValueError("Есть магазин без разрешённых маршрутов")
    return warehouses, stores, costs, routes


def np_is_integer(values) -> bool:
    """Проверяет, что все численные величины являются целыми и конечными."""
    import numpy as np
    arr = values.to_numpy(dtype=float)
    return bool(np.isfinite(arr).all() and np.allclose(arr, np.round(arr), atol=1e-9))
