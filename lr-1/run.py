"""Полный конвейер ЛР1; допустим запуск из любого рабочего каталога."""
from pathlib import Path
import json, sys, os, tempfile
import pandas as pd

LAB=Path(__file__).resolve().parent
ROOT=LAB.parent
os.environ.setdefault('MPLCONFIGDIR',str(Path(tempfile.gettempdir())/'lin-prog-lr1-matplotlib'))
sys.path.insert(0,str(ROOT))
from src.data_generation import generate_data, variant_check
from src.data_loader import load_data
from src.cutting_patterns import enumerate_patterns, save_patterns
from src.model import build_model
from src.solver import configure_log
from src.experiments import run_case, benchmark, run_sensitivity
from src.reporting import create_plots, write_report

def main():
    if ROOT.name!='lin-prog_lr1-lr3': raise RuntimeError(f'Ожидался корень lin-prog_lr1-lr3, найден {ROOT.name}')
    if not variant_check(): raise RuntimeError('Ошибка вычисления варианта или seed')
    configure_log(LAB/'logs'/'solver.log')
    generated=LAB/'data'/'generated'; generate_data(generated)
    pipes,orders=load_data(generated)
    patterns=enumerate_patterns(pipes,orders); save_patterns(patterns,generated/'cutting_patterns.csv')
    scenario_dir=LAB/'data'/'scenarios'; scenario_dir.mkdir(parents=True,exist_ok=True)
    baseline={'name':'baseline','description':'Сгенерированные базовые данные; kerf=0 мм.'}
    (scenario_dir/'baseline.json').write_text(json.dumps(baseline,ensure_ascii=False,indent=2),encoding='utf-8')
    base,bi=run_case('baseline',pipes,orders,patterns,LAB/'results'/'baseline')
    if not base.success: raise RuntimeError(f'Базовый сценарий не решён: {base.message}')
    bx=base.x
    base_usage={str(pid):sum(float(bx[k]) for k in range(len(patterns)) if str(patterns.iloc[k].pipe_id)==str(pid)) for pid in pipes.pipe_id}
    # Для содержательного shortage сокращаем на 20% тип с максимальным расходом.
    target=max(base_usage,key=base_usage.get); shortpipes=pipes.copy(); shortpipes['stock_pcs']=shortpipes.stock_pcs.astype(float); old=float(shortpipes.loc[shortpipes.pipe_id==target,'stock_pcs'].iloc[0])
    shortpipes.loc[shortpipes.pipe_id==target,'stock_pcs']=old*0.8
    shortconfig={'name':'shortage','changed_pipe_type':target,'stock_before':old,'stock_after':old*0.8,'relative_reduction':0.2}
    (scenario_dir/'shortage.json').write_text(json.dumps(shortconfig,ensure_ascii=False,indent=2),encoding='utf-8')
    short,si=run_case('shortage',shortpipes,orders,patterns,LAB/'results'/'shortage')
    # Увеличение спроса в 12 раз вместе со снижением одного запаса создаёт стресс по общей мощности.
    stresspipes=pipes.copy(); stresspipes['stock_pcs']=stresspipes.stock_pcs.astype(float); stresspipes.loc[stresspipes.pipe_id==target,'stock_pcs']=old*0.8
    stressorders=orders.copy(); stressorders['demand_pcs']=stressorders.demand_pcs*12
    stressconfig={'name':'stress','changed_pipe_type':target,'stock_before':old,'stock_after':old*0.8,'demand_multiplier':12,'orders':orders.order_id.tolist()}
    (scenario_dir/'stress.json').write_text(json.dumps(stressconfig,ensure_ascii=False,indent=2),encoding='utf-8')
    stress,ti=run_case('stress',stresspipes,stressorders,patterns,LAB/'results'/'stress')
    # Обобщённая таблица сценариев.
    summary={}; rows=[]
    for name,res,info in [('baseline',base,bi),('shortage',short,si),('stress',stress,ti)]:
        x=res.x if res.success else None
        used=float(sum(x)) if x is not None else None
        obj=float(res.fun) if res.success else None
        usage={str(pid):sum(float(x[k]) for k in range(len(patterns)) if str(patterns.iloc[k].pipe_id)==str(pid)) for pid in pipes.pipe_id} if x is not None else {}
        summary[name]={'status':info['status'],'objective':obj,'time':info['solve_time_seconds'],'used':used,'x':x if x is not None else [],'usage_by_pipe':usage}
        rows.append({'scenario':name,'status':info['status'],'objective_waste_mm':obj,'solve_time_seconds':info['solve_time_seconds'],'total_pipes_lp':used,'message':info['message']})
    summary['shortage']['changed_pipe_type']=target
    summary['shortage']['stock_before']=old
    summary['shortage']['stock_after']=float(shortpipes.loc[shortpipes.pipe_id==target,'stock_pcs'].iloc[0])
    summary['stress']['material_capacity_mm']=float((stresspipes.length_mm*stresspipes.stock_pcs).sum())
    summary['stress']['required_order_length_mm']=float((stressorders.length_mm*stressorders.demand_pcs).sum())
    pd.DataFrame(rows).to_csv(LAB/'experiments'/'scenario_results.csv',index=False)
    # Теневые цены и независимые сценарные проверки — отдельные сохраняемые артефакты.
    baseline_shadow=pd.read_csv(LAB/'results'/'baseline'/'shadow_prices.csv')
    run_sensitivity(pipes,orders,patterns,float(base.fun),baseline_shadow,LAB/'experiments')
    baseline_shadow.to_csv(LAB/'experiments'/'shadow_prices.csv',index=False)
    bench=benchmark(build_model(pipes,orders,patterns),repeats=7)
    bench.to_csv(LAB/'experiments'/'solver_benchmark.csv',index=False)
    comp=LAB/'results'/'solver_comparison'; comp.mkdir(parents=True,exist_ok=True)
    bench.to_csv(comp/'benchmark.csv',index=False)
    (comp/'summary.json').write_text(json.dumps({'repeats_per_method':7,'methods':['highs-ds','highs-ipm'],'objective_baseline':float(base.fun)},indent=2),encoding='utf-8')
    create_plots(LAB,summary,pipes,base_usage,bench)
    _,vsummary=__import__('src.validation',fromlist=['validate']).validate(base.x,pipes,orders,patterns,base.fun)
    sensitivity=pd.read_csv(LAB/'experiments'/'sensitivity_checks.csv')
    write_report(LAB,pipes,orders,patterns,summary,vsummary,baseline_shadow,sensitivity,bench,base_usage)
    print(f"Baseline: {base.fun:.8f} mm; validation={vsummary['passed']}; patterns={len(patterns)}")
    print(f"Shortage: {summary['shortage']['status']} objective={summary['shortage']['objective']}; stress: {summary['stress']['status']}")
    print(f"Report: {LAB/'reports'/'report.md'}")

if __name__=='__main__': main()
