"""Решение MILP через scipy.optimize.milp с HiGHS и журналированием."""
import logging
from pathlib import Path
from time import perf_counter
from scipy.optimize import milp

OPTIONS = {"disp": False, "presolve": True, "mip_rel_gap": 0.0}
STATUS = {0: "optimal", 1: "limit_reached", 2: "infeasible", 3: "unbounded", 4: "solver_error"}


def configure_log(path: Path) -> None:
    """Настраивает лог запуска MILP."""
    path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=path, level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s", force=True)


def solve(model, options=None):
    """Запускает HiGHS на построенной модели и измеряет только время решения."""
    actual_options = OPTIONS.copy() if options is None else options.copy()
    start = perf_counter()
    result = milp(c=model.c, integrality=model.integrality, bounds=model.bounds,
                  constraints=model.constraints, options=actual_options.copy())
    elapsed = perf_counter() - start
    logging.info("status=%s success=%s objective=%s time=%.9f nodes=%s gap=%s message=%s",
                 STATUS.get(result.status, "unknown"), result.success,
                 getattr(result, "fun", None), elapsed,
                 getattr(result, "mip_node_count", None), getattr(result, "mip_gap", None), result.message)
    return result, elapsed, actual_options


def solver_info(result, elapsed: float, options: dict) -> dict:
    """Сериализуемый статус; objective объявляется только при доказанном оптимуме."""
    has_incumbent = getattr(result, "x", None) is not None and getattr(result, "fun", None) is not None
    return {
        "status_code": int(result.status), "status": STATUS.get(result.status, "unknown"),
        "success": bool(result.success), "message": str(result.message),
        "objective_value": float(result.fun) if result.status == 0 and result.fun is not None else None,
        "incumbent_objective": float(result.fun) if has_incumbent else None,
        "solve_time_seconds": float(elapsed),
        "mip_node_count": getattr(result, "mip_node_count", None),
        "mip_gap": getattr(result, "mip_gap", None), "options": options,
    }
