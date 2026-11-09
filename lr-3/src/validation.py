"""Независимая NumPy-проверка бинарных назначений из сохранённых CSV."""
from pathlib import Path
import numpy as np
import pandas as pd

TOL = 1e-7


def validate_saved(result_dir: Path, servers: pd.DataFrame, tasks: pd.DataFrame,
                   costs: pd.DataFrame, activation_limit: int, solver_objective=None):
    """Восстанавливает x/y из CSV и пересчитывает ограничения и цель без объекта MILP."""
    values = pd.read_csv(result_dir / "variable_values.csv")
    server_ids = servers.server_id.astype(str).tolist()
    task_ids = tasks.task_id.astype(str).tolist()
    expected_count = len(server_ids) * len(task_ids) + len(server_ids)
    if len(values) != expected_count or values.duplicated(["variable_type", "server_id", "task_id"]).any():
        raise ValueError("В сохранённом решении есть пропущенные или повторяющиеся переменные")
    cost_frame = costs.pivot(index="server_id", columns="task_id", values="assignment_cost").reindex(
        index=server_ids, columns=task_ids
    )
    cost_matrix = cost_frame.to_numpy(dtype=float)
    x = np.zeros((len(server_ids), len(task_ids)), dtype=float)
    y = np.zeros(len(server_ids), dtype=float)
    wi = {key: idx for idx, key in enumerate(server_ids)}
    ti = {key: idx for idx, key in enumerate(task_ids)}
    for row in values.itertuples(index=False):
        if row.variable_type == "x":
            x[wi[str(row.server_id)], ti[str(row.task_id)]] = float(row.value)
        elif row.variable_type == "y":
            y[wi[str(row.server_id)]] = float(row.value)
        else:
            raise ValueError(f"Неизвестный тип переменной: {row.variable_type}")

    checks = []
    task_totals = x.sum(axis=0)
    for j, task_id in enumerate(task_ids):
        residual = abs(float(task_totals[j]) - 1.0)
        checks.append({"check": f"one_assignment_{task_id}", "lhs": task_totals[j], "rhs": 1,
                       "residual": residual, "passed": residual <= TOL})
    for i, server_id in enumerate(server_ids):
        assigned = float(x[i, :].sum())
        link_violation = float(max(0.0, np.max(x[i, :] - y[i])))
        orphan_violation = float(max(0.0, y[i] - assigned))
        checks.append({"check": f"assignment_requires_enabled_{server_id}", "lhs": assigned,
                       "rhs": y[i], "residual": link_violation, "passed": link_violation <= TOL})
        checks.append({"check": f"enabled_server_has_task_{server_id}", "lhs": y[i],
                       "rhs": assigned, "residual": orphan_violation, "passed": orphan_violation <= TOL})
        if int(servers.available.iloc[i]) == 0:
            residual = abs(float(y[i]))
            checks.append({"check": f"availability_{server_id}", "lhs": y[i], "rhs": 0,
                           "residual": residual, "passed": residual <= TOL})
    count_violation = max(0.0, float(y.sum()) - float(activation_limit))
    checks.append({"check": "activation_limit", "lhs": y.sum(), "rhs": activation_limit,
                   "residual": count_violation, "passed": count_violation <= TOL})
    binary_values = np.concatenate((x.ravel(), y))
    binary_error = float(np.max(np.minimum(np.abs(binary_values), np.abs(binary_values - 1.0))))
    checks.append({"check": "binary_variables", "lhs": binary_error, "rhs": 0,
                   "residual": binary_error, "passed": binary_error <= TOL})
    independent_objective = float(np.sum(cost_matrix * x))
    objective_gap = None if solver_objective is None else abs(independent_objective - float(solver_objective))
    checks.append({"check": "objective_recalculation", "lhs": independent_objective,
                   "rhs": solver_objective, "residual": objective_gap,
                   "passed": objective_gap is None or objective_gap <= TOL})
    checks_frame = pd.DataFrame(checks)

    assignment_rows = []
    for j, task_id in enumerate(task_ids):
        i = int(np.argmax(x[:, j]))
        assignment_rows.append({"task_id": task_id, "assigned_count": task_totals[j],
                                "server_id": server_ids[i], "server_enabled": y[i],
                                "assignment_cost": cost_matrix[i, j]})
    assignment = pd.DataFrame(assignment_rows)
    counts = x.sum(axis=1).astype(int)
    utilization = servers.copy()
    utilization["enabled"] = (y > 0.5).astype(int)
    utilization["assigned_tasks"] = counts
    utilization["assignment_cost"] = (cost_matrix * x).sum(axis=1)
    summary = {
        "passed": bool(checks_frame.passed.all()), "tolerance": TOL,
        "number_of_checks": int(len(checks_frame)),
        "violations": int((~checks_frame.passed).sum()),
        "max_residual": float(checks_frame.residual.fillna(0).max()),
        "independent_objective": independent_objective,
        "solver_objective": solver_objective, "objective_gap": objective_gap,
        "enabled_servers": int(np.count_nonzero(y > 0.5)),
        "assigned_tasks": int(np.count_nonzero(task_totals > 0.5)),
    }
    return checks_frame, summary, assignment, utilization, x, y
