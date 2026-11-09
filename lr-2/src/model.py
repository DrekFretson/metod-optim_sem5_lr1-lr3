"""Построение транспортной LP-модели только по разрешённым дугам."""
from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class TransportationModel:
    """Матрицы LP и соответствие каждого столбца его разрешённому маршруту."""
    c: np.ndarray
    a_ub: np.ndarray
    b_ub: np.ndarray
    a_eq: np.ndarray
    b_eq: np.ndarray
    routes: pd.DataFrame
    warehouses: pd.DataFrame
    stores: pd.DataFrame


def build_model(warehouses: pd.DataFrame, stores: pd.DataFrame,
                costs: pd.DataFrame, routes: pd.DataFrame) -> TransportationModel:
    """Строит LP-матрицы в порядке строк входных таблиц."""
    arcs = routes.loc[routes.allowed.eq(1)].merge(
        costs, on=["warehouse_id", "store_id"], validate="one_to_one"
    ).reset_index(drop=True)
    w_ids = warehouses.warehouse_id.tolist(); s_ids = stores.store_id.tolist()
    a_ub = np.zeros((len(w_ids), len(arcs)))
    a_eq = np.zeros((len(s_ids), len(arcs)))
    w_index = {key: i for i, key in enumerate(w_ids)}
    s_index = {key: j for j, key in enumerate(s_ids)}
    for k, arc in arcs.iterrows():
        a_ub[w_index[arc.warehouse_id], k] = 1
        a_eq[s_index[arc.store_id], k] = 1
    return TransportationModel(
        c=arcs.unit_cost.to_numpy(dtype=float),
        a_ub=a_ub, b_ub=warehouses.supply.to_numpy(dtype=float),
        a_eq=a_eq, b_eq=stores.demand.to_numpy(dtype=float),
        routes=arcs, warehouses=warehouses.copy(), stores=stores.copy(),
    )
