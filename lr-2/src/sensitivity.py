"""Двойственные оценки HiGHS и численная проверка их локального прогноза."""
import numpy as np
import pandas as pd
from .solver import solve


def shadow_prices(model, result, supply=None, demand=None) -> pd.DataFrame:
    """Сопоставляет маргиналы SciPy с порядком ограничений модели."""
    supply = model.b_ub if supply is None else np.asarray(supply, dtype=float)
    demand = model.b_eq if demand is None else np.asarray(demand, dtype=float)
    sent = model.a_ub @ result.x
    received = model.a_eq @ result.x
    rows = []
    for i, wh in enumerate(model.warehouses.itertuples(index=False)):
        marginal = float(result.ineqlin.marginals[i])
        active = abs(float(supply[i] - sent[i])) <= 1e-7
        interp = ("Дополнительная единица запаса уменьшает минимум на abs(marginal) в локальном диапазоне."
                  if marginal < -1e-9 else "Запас не имеет локальной ценности при текущем оптимуме.")
        rows.append({"constraint_type": "supply", "entity_id": wh.warehouse_id,
                     "marginal": marginal, "slack": float(supply[i] - sent[i]),
                     "active": active, "interpretation": interp})
    for j, shop in enumerate(model.stores.itertuples(index=False)):
        marginal = float(result.eqlin.marginals[j])
        rows.append({"constraint_type": "demand", "entity_id": shop.store_id,
                     "marginal": marginal, "slack": float(demand[j] - received[j]),
                     "active": abs(float(demand[j] - received[j])) <= 1e-7,
                     "interpretation": "Изменение затрат на единицу изменения точного спроса в локальном диапазоне."})
    return pd.DataFrame(rows)


def verify_shadow_forecasts(model, shadows: pd.DataFrame, baseline_objective: float) -> pd.DataFrame:
    """Проверяет по одному ненулевому запасу и спросу малым RHS-шагом +1."""
    checks = []
    for kind in ("supply", "demand"):
        group = shadows.loc[(shadows.constraint_type == kind) & (shadows.marginal.abs() > 1e-8)]
        if group.empty:
            checks.append({"constraint_type": kind, "entity_id": None, "rhs_delta": None,
                           "marginal": None, "predicted_delta": None, "actual_delta": None,
                           "absolute_error": None, "status": "no_nonzero_marginal_available"})
            continue
        candidates = []
        ids = model.warehouses.warehouse_id.tolist() if kind == "supply" else model.stores.store_id.tolist()
        for row in group.itertuples(index=False):
            supply = model.b_ub.copy(); demand = model.b_eq.copy()
            if kind == "supply": supply[ids.index(row.entity_id)] += 1.0
            else: demand[ids.index(row.entity_id)] += 1.0
            result, _ = solve(model, supply=supply, demand=demand)
            actual = float(result.fun - baseline_objective) if result.success else None
            predicted = float(row.marginal)
            error = abs(actual - predicted) if actual is not None else float("inf")
            candidates.append({"constraint_type": kind, "entity_id": row.entity_id, "rhs_delta": 1.0,
                               "marginal": predicted, "predicted_delta": predicted,
                               "actual_delta": actual, "absolute_error": error,
                               "status": "optimal" if result.success else "infeasible"})
        # Prefer a perturbation that remains inside the current local basis range;
        # otherwise preserve the closest real experiment and expose its error.
        best = min(candidates, key=lambda item: item["absolute_error"])
        if best["absolute_error"] > 1e-7 and best["status"] == "optimal":
            best["status"] = "optimal_outside_local_basis_or_degenerate"
        checks.append(best)
    return pd.DataFrame(checks)
