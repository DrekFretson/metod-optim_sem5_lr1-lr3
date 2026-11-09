"""Независимый пересчёт транспортного плана по CSV и сохранённому решению."""
from pathlib import Path
import numpy as np
import pandas as pd

TOL = 1e-7


def validate(plan: pd.DataFrame, warehouses: pd.DataFrame, stores: pd.DataFrame,
             costs: pd.DataFrame, routes: pd.DataFrame, solver_objective=None):
    """Пересчитывает каждый поток, запрет, остаток и затраты без OptimizeResult."""
    full = routes.merge(costs, on=["warehouse_id", "store_id"], validate="one_to_one")
    joined = full.merge(plan[["warehouse_id", "store_id", "quantity"]],
                        on=["warehouse_id", "store_id"], how="left", validate="one_to_one")
    joined["quantity"] = joined.quantity.fillna(0.0)
    joined["total_cost"] = joined.quantity * joined.unit_cost
    quantities = joined.quantity.to_numpy(dtype=float)
    unit_costs = joined.unit_cost.to_numpy(dtype=float)
    checks = []
    for _, wh in warehouses.iterrows():
        mask = joined.warehouse_id.eq(wh.warehouse_id).to_numpy()
        sent = float(np.sum(quantities[mask]))
        slack = float(wh.supply) - sent
        checks.append({"check": f"supply_{wh.warehouse_id}", "lhs": sent, "rhs": wh.supply,
                       "residual": max(0.0, -slack), "passed": slack >= -TOL})
    for _, shop in stores.iterrows():
        mask = joined.store_id.eq(shop.store_id).to_numpy()
        received = float(np.sum(quantities[mask]))
        residual = received - float(shop.demand)
        checks.append({"check": f"demand_{shop.store_id}", "lhs": received, "rhs": shop.demand,
                       "residual": abs(residual), "passed": abs(residual) <= TOL})
    min_quantity = float(np.min(quantities)) if len(quantities) else 0.0
    neg = float(max(0.0, -min_quantity))
    checks.append({"check": "nonnegativity", "lhs": min_quantity, "rhs": 0,
                   "residual": neg, "passed": neg <= TOL})
    forbidden_mask = joined.allowed.eq(0).to_numpy()
    banned_use = float(np.max(np.abs(quantities[forbidden_mask]))) if forbidden_mask.any() else 0.0
    checks.append({"check": "forbidden_routes_unused", "lhs": banned_use, "rhs": 0,
                   "residual": banned_use, "passed": banned_use <= TOL})
    independent_objective = float(np.dot(quantities, unit_costs))
    objective_gap = None if solver_objective is None else abs(independent_objective - float(solver_objective))
    checks.append({"check": "objective_recalculation", "lhs": independent_objective,
                   "rhs": solver_objective, "residual": objective_gap,
                   "passed": objective_gap is None or objective_gap <= TOL})
    result = pd.DataFrame(checks)
    summary = {
        "passed": bool(result.passed.all()), "tolerance": TOL,
        "max_residual": float(result.residual.fillna(0).max()),
        "independent_objective": independent_objective,
        "solver_objective": solver_objective, "objective_gap": objective_gap,
        "total_shipped": float(np.sum(quantities)),
        "forbidden_route_quantity": banned_use,
    }
    warehouse_use = warehouses.merge(
        joined.groupby("warehouse_id", as_index=False).quantity.sum().rename(columns={"quantity": "shipped"}),
        on="warehouse_id", how="left")
    warehouse_use["shipped"] = warehouse_use.shipped.fillna(0.0)
    warehouse_use["unused"] = warehouse_use.supply - warehouse_use.shipped
    store_service = stores.merge(
        joined.groupby("store_id", as_index=False).quantity.sum().rename(columns={"quantity": "received"}),
        on="store_id", how="left")
    store_service["received"] = store_service.received.fillna(0.0)
    store_service["difference"] = store_service.received - store_service.demand
    return result, summary, joined, warehouse_use, store_service


def validate_saved(directory: Path, warehouses, stores, costs, routes, objective):
    """Загружает сохранённый CSV решения, затем выполняет независимую проверку."""
    plan = pd.read_csv(directory / "transportation_plan.csv")
    return validate(plan, warehouses, stores, costs, routes, objective)
