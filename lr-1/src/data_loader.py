"""Загрузка и проверка табличных исходных данных."""
from pathlib import Path
import pandas as pd

def load_data(directory: Path):
    """Загружает таблицы и проверяет обязательные поля и значения."""
    try:
        pipes=pd.read_csv(directory/'pipe_types.csv'); orders=pd.read_csv(directory/'orders.csv')
    except (OSError, ValueError) as exc:
        raise ValueError(f'Не удалось прочитать исходные данные: {exc}') from exc
    for frame, cols in ((pipes,{'pipe_id','length_mm','stock_pcs'}),(orders,{'order_id','length_mm','demand_pcs'})):
        if not cols.issubset(frame.columns): raise ValueError(f'Отсутствуют столбцы: {cols-set(frame.columns)}')
        if frame[list(cols-{ 'pipe_id','order_id'})].isna().any().any() or (frame[list(cols-{ 'pipe_id','order_id'})]<=0).any().any(): raise ValueError('Длины и количества должны быть положительными')
    if len(pipes)<5 or len(orders)<6: raise ValueError('Требуется минимум 5 типов труб и 6 заказов')
    return pipes,orders
