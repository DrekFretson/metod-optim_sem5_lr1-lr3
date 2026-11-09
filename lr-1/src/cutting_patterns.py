"""Полное перечисление допустимых комбинаций резки для каждого типа трубы."""
from pathlib import Path
import pandas as pd

def enumerate_patterns(pipes: pd.DataFrame, orders: pd.DataFrame) -> pd.DataFrame:
    """Перебирает целочисленные наборы отрезков в каждой схеме без ограничения полноты."""
    rows=[]; lengths=orders.length_mm.astype(int).tolist(); ids=orders.order_id.astype(str).tolist(); serial={str(p):0 for p in pipes.pipe_id}
    for pi, pipe in pipes.iterrows():
        L=int(pipe.length_mm); counts=[0]*len(lengths)
        def visit(k, remaining):
            if k==len(lengths):
                if not any(counts): return
                used=L-remaining
                serial[str(pipe.pipe_id)]+=1
                row={'pattern_id':f'{pipe.pipe_id}_P{serial[str(pipe.pipe_id)]}', 'pipe_id':str(pipe.pipe_id), 'used_mm':used,'waste_mm':remaining}
                row.update({f'a_{oid}':counts[j] for j,oid in enumerate(ids)}); rows.append(row); return
            for n in range(remaining//lengths[k]+1):
                counts[k]=n; visit(k+1,remaining-n*lengths[k])
            counts[k]=0
        visit(0,L)
    result=pd.DataFrame(rows)
    if result.empty: raise ValueError('Допустимые схемы не найдены')
    return result

def save_patterns(patterns: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True,exist_ok=True); patterns.to_csv(path,index=False)
