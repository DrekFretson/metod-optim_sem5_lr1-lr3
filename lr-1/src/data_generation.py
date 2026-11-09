"""Воспроизводимая генерация исходных данных."""
import json, platform
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
from common.config import N, Q, Y, VARIANT, SEED

def generate_data(directory: Path) -> None:
    """Создаёт данные только при отсутствии исходных CSV."""
    directory.mkdir(parents=True, exist_ok=True)
    pfile, ofile = directory/'pipe_types.csv', directory/'orders.csv'
    if pfile.exists() and ofile.exists():
        return
    rng = np.random.default_rng(SEED)
    # Типоразмеры лежат в диапазоне 6–8.5 м, заказы — 1.2–2.2 м.
    lengths = np.sort(rng.integers(6000, 8501, size=5) // 10 * 10)
    order_lengths = np.sort(rng.integers(1200, 2201, size=6) // 10 * 10)[::-1]
    demand = rng.integers(18, 51, size=6)
    # Щедрый, но конечный запас гарантирует реализуемость набора заказов:
    # каждый отрезок можно производить отдельной схемой на самой длинной трубе.
    pipes = pd.DataFrame({'pipe_id':[f'T{i+1}' for i in range(5)], 'length_mm':lengths,
                          'stock_pcs':np.maximum(12, np.ceil(demand.sum()*order_lengths.max()/lengths).astype(int)+5)})
    orders = pd.DataFrame({'order_id':[f'O{j+1}' for j in range(6)], 'length_mm':order_lengths, 'demand_pcs':demand})
    pipes.to_csv(pfile, index=False); orders.to_csv(ofile, index=False)
    meta = {'individual_parameters':{'N':N,'Q':Q,'Y':Y,'V':VARIANT}, 'seed':SEED,
            'python_version':platform.python_version(),'numpy_version':np.__version__,
            'generated_at_utc':datetime.now(timezone.utc).isoformat(),
            'algorithm':'numpy.random.default_rng(10105); pipe lengths sampled uniformly from 6000..8500 mm and order lengths from 1200..2200 mm in 10 mm steps, then sorted; demand sampled uniformly as integers from 18..50; stock_i=max(12, ceil(sum(demand)*max(order_length)/pipe_length_i)+5).',
            'source_parameters':{'pipe_length_mm_range':[6000,8500],'order_length_mm_range':[1200,2200],'demand_pcs_range':[18,50]},
            'units':{'length':'mm','stock':'pipes','demand':'pieces'},
            'assumptions':['Kerf = 0 mm; no cutting loss beyond unused remainder.','Stocks are per type and are deliberately sufficient for baseline demand.','Continuous averaged LP; pipe quantities may be fractional.','Each order is met exactly; no overproduction is permitted.']}
    (directory/'generation_metadata.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')

def variant_check() -> bool:
    return VARIANT == 19 and SEED == 10000 + 100*Q + N and VARIANT == 1+((17*N+7*Q+Y)%20)
