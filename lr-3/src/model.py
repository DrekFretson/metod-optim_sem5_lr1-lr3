"""Построение MILP назначения задач бинарным переменным x и y."""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint
from scipy.sparse import coo_matrix


@dataclass
class AssignmentModel:
    """MILP-параметры и отображение индексов x/y в идентификаторы."""
    c: np.ndarray
    integrality: np.ndarray
    bounds: Bounds
    constraints: LinearConstraint
    servers: pd.DataFrame
    tasks: pd.DataFrame
    costs: pd.DataFrame
    activation_limit: int
    x_count: int
    variable_names: list[str]


def build_model(servers: pd.DataFrame, tasks: pd.DataFrame, costs: pd.DataFrame,
                activation_limit: int) -> AssignmentModel:
    """Создаёт вектор x(server-major, task-minor), затем y(server order)."""
    if not isinstance(activation_limit, (int, np.integer)) or activation_limit < 1:
        raise ValueError("activation_limit должен быть положительным целым числом")
    n_servers, n_tasks = len(servers), len(tasks)
    if activation_limit > n_servers:
        raise ValueError("activation_limit не может превышать число серверов")
    server_ids = servers.server_id.astype(str).tolist()
    task_ids = tasks.task_id.astype(str).tolist()
    cost_matrix = costs.pivot(index="server_id", columns="task_id", values="assignment_cost").reindex(
        index=server_ids, columns=task_ids
    ).to_numpy(dtype=float)
    if not np.isfinite(cost_matrix).all():
        raise ValueError("Матрица стоимости содержит пропуски или некорректные числа")
    x_count = n_servers * n_tasks
    n_variables = x_count + n_servers
    y_index = lambda i: x_count + i
    x_index = lambda i, j: i * n_tasks + j
    c = np.concatenate((cost_matrix.reshape(-1), np.zeros(n_servers)))
    variable_names = [f"x[{server_ids[i]},{task_ids[j]}]" for i in range(n_servers) for j in range(n_tasks)]
    variable_names += [f"y[{sid}]" for sid in server_ids]

    row_data: list[dict[int, float]] = []
    lower: list[float] = []
    upper: list[float] = []

    def add_row(values: dict[int, float], lb: float = -np.inf, ub: float = np.inf) -> None:
        row_data.append(values); lower.append(lb); upper.append(ub)

    # Ровно одно назначение каждой задачи.
    for j in range(n_tasks):
        add_row({x_index(i, j): 1.0 for i in range(n_servers)}, lb=1.0, ub=1.0)
    # Назначать задачу можно только на включённый сервер: x_ij - y_i <= 0.
    for i in range(n_servers):
        for j in range(n_tasks):
            add_row({x_index(i, j): 1.0, y_index(i): -1.0}, ub=0.0)
    # Включённый сервер обязан обслуживать хотя бы одну задачу: y_i - sum_j x_ij <= 0.
    for i in range(n_servers):
        values = {x_index(i, j): -1.0 for j in range(n_tasks)}
        values[y_index(i)] = 1.0
        add_row(values, ub=0.0)
    # Лимит числа включённых серверов.
    add_row({y_index(i): 1.0 for i in range(n_servers)}, ub=float(activation_limit))
    # Создаём доступностные ограничения только для серверов, помеченных недоступными.
    for i, available in enumerate(servers.available.to_numpy(dtype=int)):
        if available == 0:
            add_row({y_index(i): 1.0}, ub=0.0)

    rr, cc, vv = [], [], []
    for r, entries in enumerate(row_data):
        for col, value in entries.items():
            rr.append(r); cc.append(col); vv.append(value)
    matrix = coo_matrix((vv, (rr, cc)), shape=(len(row_data), n_variables)).tocsr()
    integrality = np.ones(n_variables, dtype=np.int8)
    bounds = Bounds(np.zeros(n_variables), np.ones(n_variables))
    return AssignmentModel(c, integrality, bounds,
                          LinearConstraint(matrix, np.asarray(lower), np.asarray(upper)),
                          servers.copy(), tasks.copy(), costs.copy(), int(activation_limit),
                          x_count, variable_names)
