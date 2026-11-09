"""Сценарии, устойчивость и сравнение методов HiGHS."""
import json, statistics, time
from pathlib import Path
import numpy as np
import pandas as pd
from .model import build_model
from .solver import solve, solver_info
from .validation import validate
from .sensitivity import shadow_table

def run_case(name, pipes, orders, patterns, outdir, method='highs-ds'):
    """Решает сценарий и сохраняет воспроизводимые таблицы результата."""
    outdir.mkdir(parents=True,exist_ok=True); model=build_model(pipes,orders,patterns); res,elapsed=solve(model,method)
    info=solver_info(res,elapsed,method); (outdir/'solver_info.json').write_text(json.dumps(info,ensure_ascii=False,indent=2),encoding='utf-8')
    if res.success:
        x=res.x
        pd.DataFrame({'pattern_id':patterns.pattern_id,'x_pipes':x}).to_csv(outdir/'variable_values.csv',index=False)
        pd.DataFrame([{'objective_waste_mm':res.fun,'method':method,'solve_time_seconds':elapsed}]).to_csv(outdir/'solution.csv',index=False)
        df,summary=validate(x,pipes,orders,patterns,res.fun); df.to_csv(outdir/'validation.csv',index=False)
        (outdir/'validation_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
        if name=='baseline': shadow_table(res,pipes,orders,patterns,x).to_csv(outdir/'shadow_prices.csv',index=False)
        lhs_orders={str(o.order_id):sum(float(patterns.iloc[k][f'a_{o.order_id}'])*float(x[k]) for k in range(len(x))) for _,o in orders.iterrows()}
        lhs_stock={str(p.pipe_id):sum(float(x[k]) for k in range(len(x)) if str(patterns.iloc[k].pipe_id)==str(p.pipe_id)) for _,p in pipes.iterrows()}
        pd.DataFrame([{'constraint':f'demand_{k}','lhs':v} for k,v in lhs_orders.items()]+[{'constraint':f'stock_{k}','lhs':v} for k,v in lhs_stock.items()]).to_csv(outdir/'constraint_values.csv',index=False)
    else:
        pd.DataFrame(columns=['pattern_id','x_pipes']).to_csv(outdir/'variable_values.csv',index=False)
        pd.DataFrame([{'objective_waste_mm':None,'method':method,'solve_time_seconds':elapsed}]).to_csv(outdir/'solution.csv',index=False)
        pd.DataFrame([{'constraint':'not_checked_infeasible','passed':False}]).to_csv(outdir/'validation.csv',index=False)
    return res, info

def benchmark(model, repeats=7):
    rows=[]
    for method in ('highs-ds','highs-ipm'):
        samples=[]; its=[]; objectives=[]
        for run in range(1,repeats+1):
            r,t=solve(model,method); samples.append(t); objectives.append(float(r.fun) if r.success else None); its.append(getattr(r,'nit',None))
            rows.append({'method':method,'repeat':run,'time_seconds':t,'iterations':getattr(r,'nit',None),'status':solver_info(r,t,method)['status'],'objective':r.fun if r.success else None})
        rows.append({'method':method,'repeat':'median','time_seconds':statistics.median(samples),'iterations':statistics.median([i for i in its if i is not None]) if any(i is not None for i in its) else None,'status':'summary','objective':next((v for v in objectives if v is not None),None)})
    return pd.DataFrame(rows)

def run_sensitivity(pipes,orders,patterns,base_obj,shadow,outdir):
    """Переводит RHS заказа на +1 и сравнивает линейный прогноз с пересчётом LP."""
    rows=[]
    for _,r in shadow[shadow.type=='demand equality'].iterrows():
        oid=r.constraint.removeprefix('demand_'); changed=orders.copy(); changed.loc[changed.order_id==oid,'demand_pcs']+=1
        sol,_=solve(build_model(pipes,changed,patterns)); actual=float(sol.fun-base_obj) if sol.success else None
        marginal=float(r.shadow_price); rows.append({'constraint':r.constraint,'rhs_delta':1,'shadow_price':marginal,'predicted_delta':marginal,'actual_delta':actual,'absolute_error':None if actual is None else abs(actual-marginal),'status':'optimal' if sol.success else 'infeasible'})
    pd.DataFrame(rows).to_csv(outdir/'sensitivity_checks.csv',index=False)
