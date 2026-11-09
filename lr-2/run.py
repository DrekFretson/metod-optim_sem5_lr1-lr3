"""Полный воспроизводимый запуск лабораторной работы №2."""
from pathlib import Path
import json
import os
import tempfile

# Matplotlib должен использовать writable cache вместо системного ~/.config.
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "lin-prog-lr2-mpl"))

import numpy as np
import pandas as pd

from src.data_generation import generate_data
from src.data_loader import load_data
from src.experiments import analyze_routes, choose_shortage, make_stress, run_case, solve_unrestricted
from src.model import build_model
from src.reporting import create_plots, write_report
from src.sensitivity import shadow_prices, verify_shadow_forecasts
from src.solver import configure_log
from src.validation import TOL

LAB = Path(__file__).resolve().parent


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    """Generates inputs if absent, solves scenarios, validates, analyzes and reports."""
    configure_log(LAB / "logs" / "solver.log")
    for name in ("experiments", "results/baseline", "results/shortage", "results/stress", "reports", "plots", "data/scenarios"):
        (LAB / name).mkdir(parents=True, exist_ok=True)
    generated = LAB / "data" / "generated"
    generate_data(generated)
    warehouses, stores, costs, routes = load_data(generated)
    if len(warehouses) < 4 or len(stores) < 6 or not (routes.allowed.eq(0)).any():
        raise RuntimeError("Проверки минимального размера/запрещённых маршрутов не пройдены")
    if (routes.groupby("store_id").allowed.sum() == 0).any():
        raise RuntimeError("Есть магазин без разрешённого маршрута")
    if warehouses.supply.sum() < stores.demand.sum():
        raise RuntimeError("Суммарный запас меньше суммарного спроса")

    scenarios = LAB / "data" / "scenarios"; scenarios.mkdir(parents=True, exist_ok=True)
    write_json(scenarios / "baseline.json", {"name": "baseline", "description": "Исходные запасы, спрос и сеть разрешённых маршрутов."})

    base_model, base_res, base_time, base_plan, base, base_validation = run_case(
        "baseline", warehouses, stores, costs, routes, LAB / "results" / "baseline"
    )
    if not base_res.success or not base_validation or not base_validation["passed"]:
        raise RuntimeError("Базовое решение или его независимая проверка не пройдены")

    # Map stored decision vector by its allowed route keys and build dual analysis.
    baseline_shadow = shadow_prices(base_model, base_res)
    baseline_shadow.to_csv(LAB / "experiments" / "shadow_prices.csv", index=False)
    checks = verify_shadow_forecasts(base_model, baseline_shadow, float(base_res.fun))
    checks.to_csv(LAB / "experiments" / "shadow_price_checks.csv", index=False)
    baseline_shadow.to_csv(LAB / "results" / "baseline" / "shadow_prices.csv", index=False)

    shortage_choice = choose_shortage(base_model, float(base_res.fun), baseline_shadow)
    wh_idx = shortage_choice["index"]
    shortage_id = warehouses.warehouse_id.iloc[wh_idx]
    short_warehouses = warehouses.copy(); short_warehouses["supply"] = short_warehouses.supply.astype(float)
    old_supply = float(short_warehouses.supply.iloc[wh_idx])
    new_supply = float(shortage_choice["supply"][wh_idx])
    short_warehouses.loc[short_warehouses.warehouse_id.eq(shortage_id), "supply"] = new_supply
    shortage_config = {"name": "shortage", "warehouse_id": str(shortage_id),
                       "reduction": float(shortage_choice["reduction"]),
                       "supply_before": old_supply, "supply_after": new_supply,
                       "objective_change_from_baseline": shortage_choice["cost_increase"],
                       "solver_status": "optimal" if shortage_choice["result"].success else "infeasible"}
    write_json(scenarios / "shortage.json", shortage_config)
    _, short_res, _, _, short, short_validation = run_case(
        "shortage", short_warehouses, stores, costs, routes, LAB / "results" / "shortage"
    )

    stress_warehouses, stress_stores, stress_routes, stress_config = make_stress(
        warehouses, stores, routes, base_plan
    )
    write_json(scenarios / "stress.json", stress_config)
    _, stress_res, _, _, stress, stress_validation = run_case(
        "stress", stress_warehouses, stress_stores, costs, stress_routes,
        LAB / "results" / "stress"
    )

    case_records = {"baseline": base, "shortage": short, "stress": stress}
    scenario_columns = ["scenario", "solver_status", "objective_value", "total_supply", "total_demand",
                        "used_supply", "unused_supply", "number_of_allowed_routes", "number_of_used_routes",
                        "solve_time", "independent_validation_passed"]
    pd.DataFrame([{"scenario": name, "solver_status": record["status"],
                    "objective_value": record["objective_value"], "total_supply": record["total_supply"],
                    "total_demand": record["total_demand"], "used_supply": record["used_supply"],
                    "unused_supply": record["unused_supply"],
                    "number_of_allowed_routes": record["number_of_allowed_routes"],
                    "number_of_used_routes": record["number_of_used_routes"],
                    "solve_time": record["solve_time_seconds"],
                    "independent_validation_passed": record["independent_validation_passed"]}
                   for name, record in case_records.items()], columns=scenario_columns
                  ).to_csv(LAB / "experiments" / "scenario_results.csv", index=False)

    route_analysis = analyze_routes(base_plan)
    route_analysis.to_csv(LAB / "experiments" / "route_analysis.csv", index=False)
    route_analysis.to_csv(LAB / "results" / "baseline" / "route_analysis.csv", index=False)

    # Route-closure comparison is a separate experiment, not one of the 3 scenarios.
    unrestricted_info, all_routes, unrestricted_res = solve_unrestricted(warehouses, stores, costs, routes)
    if unrestricted_info["success"]:
        unrestricted_obj = float(unrestricted_res.fun)
        all_model_routes = all_routes.merge(costs, on=["warehouse_id", "store_id"], validate="one_to_one")
        all_model = build_model(warehouses, stores, costs, all_routes)
        q = pd.Series(unrestricted_res.x, index=pd.MultiIndex.from_frame(all_model.routes[["warehouse_id", "store_id"]]))
        all_model_routes["quantity"] = [float(q.get((w, s), 0.0)) for w, s in zip(all_model_routes.warehouse_id, all_model_routes.store_id)]
        all_model_routes["total_cost"] = all_model_routes.quantity * all_model_routes.unit_cost
        all_model_routes.to_csv(LAB / "experiments" / "unrestricted_route_plan.csv", index=False)
    else:
        unrestricted_obj = None
    unrestricted = {**unrestricted_info, "objective_value": unrestricted_obj,
                    "restricted_objective_value": float(base_res.fun),
                    "cost_premium": (float(base_res.fun - unrestricted_obj) if unrestricted_obj is not None else None),
                    "allowed_routes_when_unrestricted": int(len(routes))}
    write_json(LAB / "experiments" / "unrestricted_route_comparison.json", unrestricted)

    # Additional run-time assertions implement the requested automatic checks.
    if int((routes.allowed.eq(0)).sum()) < 1 or not base_validation["passed"]:
        raise AssertionError("Проверки запретных маршрутов/независимой проверки не пройдены")
    if (base_plan.loc[base_plan.allowed.eq(0), "quantity"].abs() > TOL).any():
        raise AssertionError("Обнаружен поток по запрещённой дуге")
    if not short_res.success and short["status"] != "infeasible":
        raise AssertionError("Неожиданный статус дефицитного сценария")

    create_plots(LAB, base_plan, warehouses, stores, case_records, route_analysis)
    write_report(LAB, warehouses, stores, costs, routes, base_plan,
                 pd.read_csv(LAB / "results" / "shortage" / "transportation_plan.csv"),
                 pd.read_csv(LAB / "results" / "baseline" / "warehouse_utilization.csv"),
                 pd.read_csv(LAB / "results" / "baseline" / "store_service.csv"),
                 base_validation, base, shortage_config, stress_config, case_records,
                 baseline_shadow, checks, route_analysis, unrestricted)
    print(f"Baseline: {base['status']} objective={base['objective_value']:.6f}; validation={base_validation['passed']}")
    print(f"Routes: total={len(routes)}, allowed={int(routes.allowed.sum())}, forbidden={int((routes.allowed == 0).sum())}")
    print(f"Shortage: {short['status']} objective={short['objective_value']}; stress: {stress['status']}")
    print(f"Report: {LAB / 'reports' / 'report.md'}")


if __name__ == "__main__":
    main()
