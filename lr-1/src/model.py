"""Подготовка данных LP для scipy.optimize.linprog."""
import numpy as np
import pandas as pd

def build_model(pipes: pd.DataFrame, orders: pd.DataFrame, patterns: pd.DataFrame):
    """Возвращает c, A_eq, b_eq, A_ub, b_ub; переменные непрерывны и неотрицательны."""
    order_ids=orders.order_id.astype(str).tolist(); pipe_ids=pipes.pipe_id.astype(str).tolist()
    c=patterns.waste_mm.to_numpy(float)
    Aeq=np.array([[patterns[f'a_{oid}'].to_numpy(float) for oid in order_ids]]).reshape(len(order_ids),len(patterns))
    beq=orders.demand_pcs.to_numpy(float)
    Aub=np.array([[((patterns.pipe_id==pid).astype(float).to_numpy()) for pid in pipe_ids]]).reshape(len(pipe_ids),len(patterns))
    bub=pipes.stock_pcs.to_numpy(float)
    return c,Aeq,beq,Aub,bub
