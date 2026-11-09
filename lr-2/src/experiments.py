"""Запуск сценариев, результатов, анализа маршрутов и unrestricted-сравнения."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

from .model import build_model
from .solver import info, solve
from .validation import validate_saved


def _json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def run_case(name: str, warehouses: pd.DataFrame, stores: pd.DataFrame,
             costs: pd.DataFrame, routes: pd.DataFrame, outdir: Path):
    """Решает один набор RHS/маршрутов и сохраняет полный сценарный результат."""
    outdir.mkdir(parents=True, exist_ok=True)
    model = build_model(warehouses, stores, costs, routes)
    result, elapsed = solve(model)
    solver_info = info(result, elapsed)
    _json(outdir / "solver_info.json", solver_info)
    validation_summary = None
    if result.success:
        qty = pd.Series(result.x, index=pd.MultiIndex.from_frame(model.routes[["warehouse_id", "store_id"]]))
        plan = routes.merge(costs, on=["warehouse_id", "store_id"], validate="one_to_one")
        plan["quantity"] = [float(qty.get((w, s), 0.0)) for w, s in zip(plan.warehouse_id, plan.store_id)]
        plan["total_cost"] = plan.quantity * plan.unit_cost
        plan.to_csv(outdir / "transportation_plan.csv", index=False)
        checks, validation_summary, checked_plan, warehouse_use, store_service = validate_saved(
            outdir, warehouses, stores, costs, routes, float(result.fun)
        )
        checks.to_csv(outdir / "validation.csv", index=False)
        warehouse_use.to_csv(outdir / "warehouse_utilization.csv", index=False)
        store_service.to_csv(outdir / "store_service.csv", index=False)
        _json(outdir / "validation_summary.json", validation_summary)
        objective = float(result.fun)
        used_supply = float(warehouse_use.shipped.sum())
        used_routes = int((plan.quantity > 1e-8).sum())
    else:
        plan = routes.merge(costs, on=["warehouse_id", "store_id"], validate="one_to_one")
        plan["quantity"] = np.nan
        plan["total_cost"] = np.nan
        plan.to_csv(outdir / "transportation_plan.csv", index=False)
        pd.DataFrame([{"check": "not_run_no_optimal_solution", "passed": False}]).to_csv(outdir / "validation.csv", index=False)
        _json(outdir / "validation_summary.json", {"passed": None, "reason": "No optimal vector was returned; validation not applicable."})
        objective = None; used_supply = None; used_routes = None
    summary = {
        **solver_info, "total_supply": float(warehouses.supply.sum()),
        "total_demand": float(stores.demand.sum()), "used_supply": used_supply,
        "unused_supply": (float(warehouses.supply.sum() - used_supply) if used_supply is not None else None),
        "number_of_allowed_routes": int(routes.allowed.sum()),
        "number_of_used_routes": used_routes,
        "independent_validation_passed": validation_summary["passed"] if validation_summary else None,
    }
    _json(outdir / "solution_summary.json", summary)
    return model, result, elapsed, plan, summary, validation_summary


def choose_shortage(base_model, base_objective: float, shadow: pd.DataFrame):
    """Подбирает минимальное содержательное сокращение запаса, повторно проверяя LP."""
    supply_shadow = shadow.loc[(shadow.constraint_type == "supply")].copy()
    supply_shadow["scarcity"] = -supply_shadow.marginal
    ranked_ids = supply_shadow.sort_values("scarcity", ascending=False).entity_id.tolist()
    candidates = [base_model.warehouses.warehouse_id.tolist().index(w) for w in ranked_ids]
    candidates += [i for i in np.argsort(-base_model.b_ub) if i not in candidates]
    trials = []
    for reduction in (0.20, 0.25, 0.30, 0.40, 0.50, 0.65):
        for idx in candidates:
            supply = base_model.b_ub.copy(); supply[idx] *= 1 - reduction
            result, elapsed = solve(base_model, supply=supply)
            record = {"index": int(idx), "reduction": reduction, "supply": supply,
                      "result": result, "elapsed": elapsed,
                      "cost_increase": float(result.fun - base_objective) if result.success else None}
            trials.append(record)
            if result.success and result.fun > base_objective + 1e-7:
                return record
    # A physically infeasible resource-shortage is a valid scenario if no feasible
    # tested reduction changes cost; select the mildest such case and explain it.
    infeasible = [r for r in trials if not r["result"].success]
    if infeasible:
        return infeasible[0]
    return max(trials, key=lambda r: r["cost_increase"] or 0.0)


def make_stress(warehouses, stores, routes, baseline_plan):
    """Сокращает запасы двух занятых складов, повышает спрос двух магазинов,
    и блокирует используемый маршрут только если остаётся альтернатива.
    """
    wh = warehouses.copy(); wh["supply"] = wh.supply.astype(float)
    store = stores.copy(); store["demand"] = store.demand.astype(float)
    used_by_wh = baseline_plan.loc[baseline_plan.allowed.eq(1)].groupby("warehouse_id").quantity.sum()
    critical_wh = used_by_wh.sort_values(ascending=False).head(2).index.tolist()
    for wid in critical_wh:
        wh.loc[wh.warehouse_id.eq(wid), "supply"] *= 0.70
    used_by_store = baseline_plan.loc[baseline_plan.allowed.eq(1)].groupby("store_id").quantity.sum()
    critical_stores = used_by_store.sort_values(ascending=False).head(2).index.tolist()
    for sid in critical_stores:
        store.loc[store.store_id.eq(sid), "demand"] *= 1.20
    changed_routes = routes.copy()
    candidates = baseline_plan.loc[(baseline_plan.allowed.eq(1)) & (baseline_plan.quantity > 1e-8)]
    blocked = None
    for row in candidates.sort_values("quantity", ascending=False).itertuples(index=False):
        alternatives = changed_routes.loc[(changed_routes.store_id == row.store_id) &
                                          (changed_routes.allowed.eq(1)) &
                                          (changed_routes.warehouse_id != row.warehouse_id)]
        if not alternatives.empty:
            blocked = (row.warehouse_id, row.store_id)
            changed_routes.loc[(changed_routes.warehouse_id == row.warehouse_id) &
                               (changed_routes.store_id == row.store_id), "allowed"] = 0
            break
    config = {"name": "stress",
              "warehouse_supply_reduction": {wid: {"before": float(warehouses.loc[warehouses.warehouse_id.eq(wid), "supply"].iloc[0]),
                                                     "after": float(wh.loc[wh.warehouse_id.eq(wid), "supply"].iloc[0]), "fraction": 0.30} for wid in critical_wh},
              "demand_increase": {sid: {"before": float(stores.loc[stores.store_id.eq(sid), "demand"].iloc[0]),
                                         "after": float(store.loc[store.store_id.eq(sid), "demand"].iloc[0]), "fraction": 0.20} for sid in critical_stores},
              "additional_forbidden_route": ({"warehouse_id": blocked[0], "store_id": blocked[1]} if blocked else None)}
    return wh, store, changed_routes, config


def analyze_routes(plan: pd.DataFrame) -> pd.DataFrame:
    """Добавляет использование и долю затрат к каждой потенциальной дуге."""
    result = plan.copy()
    total = float(result.total_cost.sum(skipna=True))
    result["used"] = result.quantity.fillna(0).gt(1e-8)
    result["share_of_total_transport_cost"] = result.total_cost.fillna(0) / total if total else 0.0
    return result


def solve_unrestricted(warehouses, stores, costs, routes):
    """Сравнивает baseline с теми же данными, временно разрешив все дуги."""
    all_routes = routes.copy(); all_routes["allowed"] = 1
    model = build_model(warehouses, stores, costs, all_routes)
    result, elapsed = solve(model)
    return info(result, elapsed), all_routes, result
