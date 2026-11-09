"""Извлечение маргинальных цен и локальная проверка сценариями."""
import pandas as pd

def shadow_table(res, pipes, orders, patterns, x, tol=1e-7):
    """Формирует ограничения и оценки HiGHS; linprog маргиналы приведены для RHS."""
    rows=[]
    for i,(_,o) in enumerate(orders.iterrows()):
        lhs=sum(float(patterns.iloc[k][f'a_{o.order_id}'])*float(x[k]) for k in range(len(x)))
        marg=float(res.eqlin.marginals[i]) if hasattr(res,'eqlin') else None
        rows.append({'constraint':f'demand_{o.order_id}','type':'demand equality','rhs':o.demand_pcs,'lhs':lhs,'slack':float(o.demand_pcs)-lhs,'active':abs(float(o.demand_pcs)-lhs)<=tol,'shadow_price':marg,'rhs_change_convention':'objective change ≈ marginal × RHS change'})
    for i,(_,p) in enumerate(pipes.iterrows()):
        lhs=sum(float(x[k]) for k in range(len(x)) if str(patterns.iloc[k].pipe_id)==str(p.pipe_id)); slack=float(p.stock_pcs)-lhs
        marg=float(res.ineqlin.marginals[i]) if hasattr(res,'ineqlin') else None
        rows.append({'constraint':f'stock_{p.pipe_id}','type':'stock upper bound','rhs':p.stock_pcs,'lhs':lhs,'slack':slack,'active':abs(slack)<=tol,'shadow_price':marg,'rhs_change_convention':'objective change ≈ marginal × RHS change'})
    return pd.DataFrame(rows)
