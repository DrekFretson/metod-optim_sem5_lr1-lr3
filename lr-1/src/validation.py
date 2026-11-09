"""Независимая проверка результата по исходным таблицам и вектору переменных."""
import numpy as np
import pandas as pd

EPS=1e-7
def validate(x, pipes, orders, patterns, solver_objective=None):
    """Пересчитывает все ограничения независимо от построенной LP-матрицы."""
    out=[]; objective=0.0; maxdev=0.0
    for k,(_,p) in enumerate(patterns.iterrows()):
        objective += float(p.waste_mm)*float(x[k])
    for _,o in orders.iterrows():
        lhs=sum(float(patterns.iloc[k][f'a_{o.order_id}'])*float(x[k]) for k in range(len(x)))
        dev=abs(lhs-float(o.demand_pcs)); maxdev=max(maxdev,dev)
        out.append({'constraint':f'demand_{o.order_id}','type':'equality','rhs':o.demand_pcs,'lhs':lhs,'slack':float(o.demand_pcs)-lhs,'max_deviation':dev,'passed':dev<=EPS})
    for _,p in pipes.iterrows():
        lhs=sum(float(x[k]) for k in range(len(x)) if str(patterns.iloc[k].pipe_id)==str(p.pipe_id))
        slack=float(p.stock_pcs)-lhs; dev=max(0.0,-slack); maxdev=max(maxdev,dev)
        out.append({'constraint':f'stock_{p.pipe_id}','type':'capacity','rhs':p.stock_pcs,'lhs':lhs,'slack':slack,'max_deviation':dev,'passed':dev<=EPS})
    minx=float(np.min(x)) if len(x) else float('nan'); out.append({'constraint':'nonnegativity','type':'bound','rhs':0,'lhs':minx,'slack':minx,'max_deviation':max(0,-minx),'passed':minx>=-EPS})
    gap=None if solver_objective is None else abs(objective-float(solver_objective))
    out.append({'constraint':'objective_recalculation','type':'objective','rhs':solver_objective,'lhs':objective,'slack':None,'max_deviation':gap,'passed':gap is None or gap<=EPS})
    df=pd.DataFrame(out); return df, {'passed':bool(df.passed.all()),'tolerance':EPS,'max_deviation':maxdev,'objective_recalculated':objective,'objective_solver':solver_objective,'objective_gap':gap}
