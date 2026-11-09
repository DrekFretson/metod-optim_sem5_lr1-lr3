"""Решение LP средствами SciPy/HiGHS и журналирование."""
import json, logging, time
from pathlib import Path
from scipy.optimize import linprog

def configure_log(path: Path):
    path.parent.mkdir(parents=True,exist_ok=True)
    logging.basicConfig(filename=path,level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s',force=True)

def solve(model, method='highs-ds', options=None):
    """Решает модель и возвращает результат плюс измеренное время."""
    c,aeq,beq,aub,bub=model; start=time.perf_counter()
    res=linprog(c,A_eq=aeq,b_eq=beq,A_ub=aub,b_ub=bub,bounds=(0,None),method=method,options=options or {})
    elapsed=time.perf_counter()-start
    logging.info('method=%s status=%s success=%s objective=%s time=%.9f message=%s',method,res.status,res.success,res.fun,elapsed,res.message)
    return res,elapsed

def solver_info(res, elapsed, method):
    return {'method':method,'status_code':int(res.status),'status':{0:'optimal',1:'iteration_limit',2:'infeasible',3:'unbounded',4:'solver_error'}.get(res.status,'unknown'),
            'success':bool(res.success),'message':str(res.message),'objective':None if res.fun is None else float(res.fun),
            'solve_time_seconds':elapsed,'iterations':getattr(res,'nit',None)}

def write_info(path: Path, info): path.write_text(json.dumps(info,ensure_ascii=False,indent=2),encoding='utf-8')
