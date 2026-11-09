"""Воспроизводимая генерация серверов, задач и матрицы назначения."""
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import scipy

SEED = 10105
SERVER_COUNT = 8
TASK_COUNT = 16
BASE_K = 5


def highs_version() -> str:
    """Получает версию HiGHS из SciPy, если она экспортируется данной сборкой."""
    try:
        from scipy.optimize._highspy import _core
        return ".".join(str(getattr(_core, key)) for key in
                         ("HIGHS_VERSION_MAJOR", "HIGHS_VERSION_MINOR", "HIGHS_VERSION_PATCH"))
    except (ImportError, AttributeError):
        return "bundled with SciPy; version not exposed by public API"


def generate_data(directory: Path) -> None:
    """Генерирует исходные CSV при их отсутствии и записывает метаданные."""
    directory.mkdir(parents=True, exist_ok=True)
    names = ("servers.csv", "tasks.csv", "assignment_costs.csv", "generation_metadata.json")
    if all((directory / name).is_file() for name in names):
        return
    rng = np.random.default_rng(SEED)
    server_ids = [f"SRV{i+1:02d}" for i in range(SERVER_COUNT)]
    task_ids = [f"TASK{i+1:02d}" for i in range(TASK_COUNT)]
    servers = pd.DataFrame({
        "server_id": server_ids,
        "server_name": [f"Вычислительный сервер {i+1}" for i in range(SERVER_COUNT)],
        "available": np.ones(SERVER_COUNT, dtype=int),
    })
    tasks = pd.DataFrame({
        "task_id": task_ids,
        "task_name": [f"Вычислительная задача {i+1}" for i in range(TASK_COUNT)],
    })

    # Каждая машина является предпочтительной ровно для двух задач до
    # перемешивания. Остальные пары получают положительные разные тарифы.
    preferred = np.repeat(np.arange(SERVER_COUNT), TASK_COUNT // SERVER_COUNT)
    rng.shuffle(preferred)
    costs = rng.integers(28, 76, size=(SERVER_COUNT, TASK_COUNT))
    for j, i in enumerate(preferred):
        costs[i, j] = int(rng.integers(8, 20))
    # Случайная матрица содержит дифференцированные расходы и конкурентные
    # альтернативы; все серверы остаются доступными в базовом наборе данных.
    cost_rows = [
        {"server_id": server_ids[i], "task_id": task_ids[j], "assignment_cost": int(costs[i, j])}
        for i in range(SERVER_COUNT) for j in range(TASK_COUNT)
    ]
    servers.to_csv(directory / "servers.csv", index=False)
    tasks.to_csv(directory / "tasks.csv", index=False)
    pd.DataFrame(cost_rows).to_csv(directory / "assignment_costs.csv", index=False)
    metadata = {
        "N": 5, "Q": 1, "Y": 26, "group": 517,
        "variant_formula": "1 + ((17*N + 7*Q + Y) mod 20)",
        "variant": 1 + ((17 * 5 + 7 * 1 + 26) % 20), "seed": SEED,
        "python_version": platform.python_version(), "numpy_version": np.__version__,
        "pandas_version": pd.__version__, "scipy_version": scipy.__version__,
        "highs_version": highs_version(),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "server_count": SERVER_COUNT, "task_count": TASK_COUNT,
        "baseline_activation_limit_K": BASE_K,
        "assignment_cost_range": [8, 75],
        "algorithm": "numpy.random.default_rng(10105); integer positive assignment costs; each server is preselected as the preferred server for two shuffled tasks, with preferred costs 8..19 and other costs 28..75; all servers available; no fixed activation cost or resource capacities.",
        "units": {"assignment_cost": "conditional monetary units per task", "activation_cost": "not used", "K": "servers"},
        "assumptions": [
            "Each task is indivisible and must be assigned to exactly one server.",
            "Servers have no task-count, memory, CPU, or throughput capacity constraints.",
            "All servers are available in the generated baseline data.",
            "Only the number of enabled servers is limited; activation has no fixed cost.",
            "Assignment variables and enabled-server variables are binary.",
        ],
    }
    (directory / "generation_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
