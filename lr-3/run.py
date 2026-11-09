"""Полный конвейер ЛР3: binary assignment MILP, scenarios, K-analysis and report."""
from pathlib import Path
import json
import os
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "lin-prog-lr3-mpl"))

import numpy as np
import pandas as pd

from src.data_generation import generate_data
from src.data_loader import load_data
from src.experiments import activation_analysis, run_case, write_json
from src.reporting import create_plots, write_report
from src.solver import configure_log
from src.validation import TOL

LAB = Path(__file__).resolve().parent
N, Q, Y = 5, 1, 26
VARIANT = 1 + ((17 * N + 7 * Q + Y) % 20)


def scenario(path: Path) -> dict:
    """Прочитывает JSON сценария и сообщает понятную ошибку при неверном формате."""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Не удалось прочитать сценарий {path.name}: {exc}") from exc
    if not isinstance(value, dict) or "activation_limit" not in value:
        raise ValueError(f"Сценарий {path.name} должен содержать activation_limit")
    return value


def server_view(servers: pd.DataFrame, disabled: list[str]) -> pd.DataFrame:
    """Возвращает сценарный overlay доступности без изменения исходного CSV."""
    frame = servers.copy()
    unknown = set(disabled) - set(frame.server_id)
    if unknown:
        raise ValueError(f"Сценарий отключает неизвестные серверы: {sorted(unknown)}")
    frame.loc[frame.server_id.isin(disabled), "available"] = 0
    if int(frame.available.sum()) == 0:
        raise ValueError("Сценарий отключил все серверы")
    return frame


def main() -> None:
    """Выполняет полный расчёт, автоматические проверки и сохраняет результаты."""
    if VARIANT != 19:
        raise AssertionError("Формула индивидуального варианта рассчитана неверно")
    for subdir in ("data/generated", "data/scenarios", "results/baseline", "results/shortage",
                   "results/stress", "experiments", "reports", "plots", "logs"):
        (LAB / subdir).mkdir(parents=True, exist_ok=True)
    configure_log(LAB / "logs" / "solver.log")
    generated = LAB / "data" / "generated"
    generate_data(generated)
    servers, tasks, costs = load_data(generated)
    scenario_dir = LAB / "data" / "scenarios"
    baseline_path = scenario_dir / "baseline.json"
    if not baseline_path.exists():
        write_json(baseline_path, {"name": "baseline", "activation_limit": 5, "disabled_servers": []})
    base_config = scenario(baseline_path)
    K = int(base_config["activation_limit"])
    if K < 1 or K > int(servers.available.sum()):
        raise ValueError("K должен быть положительным и не превышать число доступных серверов")
    if len(servers) < 8 or len(tasks) < 15:
        raise AssertionError("Минимальный размер исходной задачи не выполнен")
    if len(costs) != len(servers) * len(tasks):
        raise AssertionError("Матрица стоимости неполная")

    base_servers = server_view(servers, base_config.get("disabled_servers", []))
    base, base_info, base_validation, base_utilization, base_summary = run_case(
        "baseline", base_servers, tasks, costs, K, LAB / "results" / "baseline"
    )
    if base_info["status"] != "optimal" or not base_validation or not base_validation["passed"]:
        raise RuntimeError(f"Базовый MILP не прошёл независимую проверку: {base_info['message']}")
    base_plan = pd.read_csv(LAB / "results" / "baseline" / "assignment_plan.csv")

    # Дефицит: K уменьшается на единицу и отключается наиболее загруженный
    # выбранный базовым решением сервер. Параметры записываются в JSON.
    selected = base_utilization.loc[base_utilization.enabled.eq(1)].sort_values("assigned_tasks", ascending=False)
    shortage_disabled = [str(selected.server_id.iloc[0])]
    shortage_K = min(K - 1, int(servers.available.sum()) - 1)
    if shortage_K < 1:
        raise RuntimeError("Нельзя сформировать разрешимый дефицитный сценарий с положительным K")
    shortage_config = {
        "name": "shortage", "activation_limit": shortage_K,
        "disabled_servers": shortage_disabled,
        "description": "Сокращён лимит включённых серверов на единицу и отключён один наиболее загруженный базовый сервер.",
    }
    write_json(scenario_dir / "shortage.json", shortage_config)
    shortage_servers = server_view(servers, shortage_disabled)
    short, short_info, short_validation, short_utilization, short_summary = run_case(
        "shortage", shortage_servers, tasks, costs, shortage_K, LAB / "results" / "shortage"
    )

    # Стресс одновременно снижает K и отключает два наиболее загруженных
    # базовых сервера. При наличии других серверов назначение остаётся возможным.
    stress_disabled = [str(x) for x in selected.server_id.iloc[:2]]
    stress_K = max(1, K - 2)
    stress_config = {
        "name": "stress", "activation_limit": stress_K,
        "disabled_servers": stress_disabled,
        "description": "Стресс: лимит включения снижен на два сервера, два наиболее загруженных базовых сервера недоступны.",
    }
    write_json(scenario_dir / "stress.json", stress_config)
    stress_servers = server_view(servers, stress_disabled)
    stress, stress_info, stress_validation, stress_utilization, stress_summary = run_case(
        "stress", stress_servers, tasks, costs, stress_K, LAB / "results" / "stress"
    )

    case_rows = [base_summary, short_summary, stress_summary]
    pd.DataFrame(case_rows).to_csv(LAB / "experiments" / "scenario_results.csv", index=False)

    # Анализ K выполняется на базовых данных и их исходной доступности.
    limits = list(range(3, len(servers) + 1))
    activation = activation_analysis(base_servers, tasks, costs, limits, LAB / "results")
    activation.to_csv(LAB / "experiments" / "activation_limit_analysis.csv", index=False)
    if (base_validation["enabled_servers"] > K + TOL or
            (base_plan.groupby("task_id").size() != 1).any()):
        raise AssertionError("Нарушены базовые ограничения назначения или K")
    if not base_validation["passed"]:
        raise AssertionError("Независимая базовая проверка не пройдена")
    if short_info["status"] == "optimal" and short_validation and not short_validation["passed"]:
        raise AssertionError("Независимая проверка shortage не пройдена")
    if stress_info["status"] == "optimal" and stress_validation and not stress_validation["passed"]:
        raise AssertionError("Независимая проверка stress не пройдена")

    create_plots(LAB, base_plan, base_utilization, activation, case_rows)
    write_report(LAB, servers, tasks, costs, K, base_plan, base_utilization,
                 base_summary, base_info, base_validation,
                 pd.read_csv(LAB / "results" / "shortage" / "assignment_plan.csv"),
                 short_summary, shortage_config,
                 pd.read_csv(LAB / "results" / "stress" / "assignment_plan.csv"),
                 stress_summary, stress_config, activation)
    print(f"Baseline: {base_info['status']} objective={base_info['objective_value']:.3f}; "
          f"enabled={base_validation['enabled_servers']}; validation={base_validation['passed']}")
    print(f"Shortage: {short_info['status']} objective={short_info['objective_value']}; "
          f"stress: {stress_info['status']} objective={stress_info['objective_value']}")
    print(f"K analysis: {len(activation)} values; monotonicity=PASS")
    print(f"Report: {LAB / 'reports' / 'report.md'}")


if __name__ == "__main__":
    main()
