"""Загрузка и валидация входных CSV для MILP назначения."""
from pathlib import Path
import numpy as np
import pandas as pd


def load_data(directory: Path):
    """Загружает таблицы, проверяет число элементов, ключи и полный набор затрат."""
    try:
        servers = pd.read_csv(directory / "servers.csv")
        tasks = pd.read_csv(directory / "tasks.csv")
        costs = pd.read_csv(directory / "assignment_costs.csv")
    except (OSError, ValueError) as exc:
        raise ValueError(f"Ошибка чтения исходных данных: {exc}") from exc
    required = ((servers, {"server_id", "server_name", "available"}),
                (tasks, {"task_id", "task_name"}),
                (costs, {"server_id", "task_id", "assignment_cost"}))
    for frame, columns in required:
        missing = columns - set(frame.columns)
        if missing:
            raise ValueError(f"Не хватает обязательных столбцов: {sorted(missing)}")
        if frame[list(columns)].isna().any().any():
            raise ValueError("Исходные CSV содержат пустые обязательные поля")
    if len(servers) < 8 or len(tasks) < 15:
        raise ValueError("Требуется не менее 8 серверов и 15 задач")
    if servers.server_id.duplicated().any() or tasks.task_id.duplicated().any():
        raise ValueError("Идентификаторы серверов и задач должны быть уникальными")
    if costs.duplicated(["server_id", "task_id"]).any():
        raise ValueError("Есть повторяющиеся пары сервер–задача")
    servers["available"] = pd.to_numeric(servers.available, errors="coerce")
    costs["assignment_cost"] = pd.to_numeric(costs.assignment_cost, errors="coerce")
    if not servers.available.isin([0, 1]).all():
        raise ValueError("servers.available должен содержать только 0 или 1")
    values = costs.assignment_cost.to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("Стоимость назначения должна быть положительной и конечной")
    expected_pairs = len(servers) * len(tasks)
    if len(costs) != expected_pairs:
        raise ValueError("Нужна стоимость для каждой пары сервер–задача")
    if set(costs.server_id) != set(servers.server_id) or set(costs.task_id) != set(tasks.task_id):
        raise ValueError("В таблице затрат есть неизвестный сервер или задача")
    if int(servers.available.sum()) < 1:
        raise ValueError("Нет доступных серверов")
    return servers.reset_index(drop=True), tasks.reset_index(drop=True), costs.reset_index(drop=True)
