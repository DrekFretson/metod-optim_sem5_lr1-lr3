"""Вызов scipy.optimize.linprog/HiGHS, лог и нормализация статусов."""
import logging
from pathlib import Path
from time import perf_counter
from scipy.optimize import linprog

STATUS = {0: "optimal", 1: "limit_reached", 2: "infeasible", 3: "unbounded", 4: "solver_error"}


def configure_log(path: Path) -> None:
    """Настраивает файл лога для каждого полного запуска."""
    path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=path, level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s", force=True)


def solve(model, *, supply=None, demand=None, method="highs"):
    """Решает LP; по умолчанию правые части берутся из модели."""
    b_ub = model.b_ub if supply is None else supply
    b_eq = model.b_eq if demand is None else demand
    start = perf_counter()
    result = linprog(model.c, A_ub=model.a_ub, b_ub=b_ub,
                     A_eq=model.a_eq, b_eq=b_eq,
                     bounds=(0, None), method=method)
    elapsed = perf_counter() - start
    logging.info("method=%s status=%s success=%s objective=%s time=%.9f message=%s",
                 method, STATUS.get(result.status, "unknown"), result.success,
                 getattr(result, "fun", None), elapsed, result.message)
    return result, elapsed


def info(result, elapsed: float, method="highs") -> dict:
    """Сериализуемая сводка решателя, не читает fun для неуспешного статуса."""
    return {
        "method": method, "status_code": int(result.status),
        "status": STATUS.get(result.status, "unknown"), "success": bool(result.success),
        "message": str(result.message),
        "objective_value": float(result.fun) if result.success else None,
        "iterations": getattr(result, "nit", None), "solve_time_seconds": elapsed,
    }
