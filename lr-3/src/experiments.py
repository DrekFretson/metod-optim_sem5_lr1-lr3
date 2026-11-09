"""Выполнение базового/сценарных MILP и эксперимента по лимиту K."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

from .model import build_model
from .solver import solve, solver_info
from .validation import validate_saved


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def run_case(name: str, servers: pd.DataFrame, tasks: pd.DataFrame, costs: pd.DataFrame,
             activation_limit: int, result_dir: Path):
    """Строит, решает и сохраняет все переменные/назначения/проверки одного сценария."""
    result_dir.mkdir(parents=True, exist_ok=True)
    model = build_model(servers, tasks, costs, activation_limit)
    result, elapsed, options = solve(model)
    info = solver_info(result, elapsed, options)
    write_json(result_dir / "solver_info.json", info)
    if result.status == 0 and result.x is not None:
        values = []
        n_servers, n_tasks = len(servers), len(tasks)
        x = result.x[:model.x_count].reshape(n_servers, n_tasks)
        y = result.x[model.x_count:]
        for i, server in enumerate(servers.itertuples(index=False)):
            for j, task in enumerate(tasks.itertuples(index=False)):
                values.append({"variable_type": "x", "server_id": server.server_id,
                               "task_id": task.task_id, "value": float(x[i, j])})
        for i, server in enumerate(servers.itertuples(index=False)):
            values.append({"variable_type": "y", "server_id": server.server_id,
                           "task_id": "", "value": float(y[i])})
        pd.DataFrame(values).to_csv(result_dir / "variable_values.csv", index=False)
        checks, validation, assignment, utilization, _, _ = validate_saved(
            result_dir, servers, tasks, costs, activation_limit, float(result.fun)
        )
        assignment = assignment.merge(tasks[["task_id", "task_name"]], on="task_id", validate="one_to_one")
        assignment = assignment.merge(servers[["server_id", "server_name"]], on="server_id", validate="many_to_one")
        assignment = assignment[["task_id", "task_name", "server_id", "server_name", "assignment_cost", "server_enabled", "assigned_count"]]
        assignment.to_csv(result_dir / "assignment_plan.csv", index=False)
        enabled = utilization.loc[utilization.enabled.eq(1)].copy()
        disabled = utilization.loc[utilization.enabled.eq(0)].copy()
        utilization.to_csv(result_dir / "server_utilization.csv", index=False)
        enabled.to_csv(result_dir / "enabled_servers.csv", index=False)
        checks.to_csv(result_dir / "validation.csv", index=False)
        write_json(result_dir / "validation_summary.json", validation)
        write_json(result_dir / "solution_summary.json", {
            **info, "activation_limit": int(activation_limit),
            "enabled_server_ids": enabled.server_id.tolist(),
            "disabled_server_ids": disabled.server_id.tolist(),
            "total_assignment_cost": float(result.fun), "activation_cost": 0.0,
            "number_of_enabled_servers": int(enabled.shape[0]),
            "remaining_activation_slots": int(activation_limit - enabled.shape[0]),
            "validation_passed": validation["passed"],
        })
    else:
        pd.DataFrame(columns=["variable_type", "server_id", "task_id", "value"]).to_csv(result_dir / "variable_values.csv", index=False)
        pd.DataFrame(columns=["task_id", "task_name", "server_id", "server_name", "assignment_cost", "server_enabled"]).to_csv(result_dir / "assignment_plan.csv", index=False)
        pd.DataFrame([{"check": "not_run_no_proven_optimum", "passed": False}]).to_csv(result_dir / "validation.csv", index=False)
        write_json(result_dir / "validation_summary.json", {"passed": None, "reason": "No proven optimal solution returned."})
        validation = None; utilization = servers.assign(enabled=0, assigned_tasks=0, assignment_cost=0.0)
        write_json(result_dir / "solution_summary.json", {**info, "activation_limit": int(activation_limit), "validation_passed": None})
    summary = {
        "scenario": name, "solver_status": info["status"],
        "objective_value": info["objective_value"], "number_of_servers": len(servers),
        "number_of_tasks": len(tasks), "activation_limit": int(activation_limit),
        "enabled_servers": (int(validation["enabled_servers"]) if validation else None),
        "disabled_servers": (len(servers) - int(validation["enabled_servers"]) if validation else None),
        "total_assignment_cost": info["objective_value"], "total_activation_cost": 0.0,
        "solve_time": elapsed, "mip_gap": info["mip_gap"],
        "independent_validation_passed": validation["passed"] if validation else None,
    }
    return result, info, validation, utilization, summary


def activation_analysis(servers, tasks, costs, values, results_root: Path):
    """Решает базовую модель для каждого K и independently validates saved output."""
    rows = []
    for k in values:
        record = run_case(f"K={k}", servers, tasks, costs, int(k), results_root / "activation_limit" / f"K_{k}")
        _, info, validation, _, summary = record
        rows.append({"activation_limit": int(k), "solver_status": info["status"],
                     "objective_value": info["objective_value"],
                     "enabled_servers": validation["enabled_servers"] if validation else None,
                     "solve_time": info["solve_time_seconds"], "mip_gap": info["mip_gap"],
                     "independent_validation_passed": validation["passed"] if validation else None})
    frame = pd.DataFrame(rows)
    feasible = frame.loc[frame.solver_status.eq("optimal")].sort_values("activation_limit")
    if not (np.diff(feasible.objective_value.to_numpy(dtype=float)) <= 1e-7).all():
        raise AssertionError("Оптимальная стоимость ухудшилась при увеличении K")
    return frame
