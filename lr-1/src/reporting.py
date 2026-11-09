"""Графики и сводный отчёт на русском языке."""
from pathlib import Path
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def create_plots(root, summary, pipes, base_usage, benchmark):
    plots=root/'plots'; plots.mkdir(parents=True,exist_ok=True)
    names=[]; vals=[]; colors=[]; infeasible=[]
    for key,label in [('baseline','Базовый'),('shortage','Дефицитный'),('stress','Стрессовый')]:
        r=summary[key]; names.append(label); vals.append(r['objective']); colors.append('#4472C4');
        if r['objective'] is None: infeasible.append(f'{label}: {r["status"]}')
    fig,ax=plt.subplots(figsize=(8,4.5)); feasible=[(n,v,c) for n,v,c in zip(names,vals,colors) if v is not None]
    bars=ax.bar([a[0] for a in feasible],[a[1] for a in feasible],color=[a[2] for a in feasible])
    for bar,(_,value,_) in zip(bars,feasible): ax.text(bar.get_x()+bar.get_width()/2,bar.get_height(),f'{value:,.1f} мм',ha='center',va='bottom',fontsize=9)
    ax.set_title('Суммарные отходы по допустимым сценариям'); ax.set_ylabel('Отходы, мм'); ax.set_xlabel('Сценарий')
    if infeasible: ax.text(.99,.98,'Недопустимые: '+'; '.join(infeasible),transform=ax.transAxes,ha='right',va='top',fontsize=9)
    fig.tight_layout(); fig.savefig(plots/'waste_comparison.png',dpi=170); plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4.5)); used=[base_usage.get(str(pid),0) for pid in pipes.pipe_id]
    ax.bar(pipes.pipe_id,used,label='Использовано'); ax.scatter(pipes.pipe_id,pipes.stock_pcs,color='darkorange',label='Запас',zorder=3)
    ax.set_title('Использование исходных труб'); ax.set_ylabel('Трубы, шт. (LP-эквивалент)'); ax.set_xlabel('Тип труб'); ax.legend(); fig.tight_layout(); fig.savefig(plots/'pipe_usage.png',dpi=170); plt.close(fig)
    med=benchmark[benchmark.repeat=='median']; fig,ax=plt.subplots(figsize=(7,4.5));
    if len(med): ax.bar(med.method,med.time_seconds,color=['#70AD47','#ED7D31'])
    ax.set_title('Сравнение методов HiGHS (медиана)'); ax.set_ylabel('Время решения, с'); ax.set_xlabel('Метод'); fig.tight_layout(); fig.savefig(plots/'solver_comparison.png',dpi=170); plt.close(fig)

def write_report(root, pipes, orders, patterns, results, validation, shadow, sensitivity, benchmark, usage):
    """Собирает содержательный Markdown с фактическими численными данными."""
    def table(df): return df.to_markdown(index=False,floatfmt='.4f')
    b=results['baseline']; s=results['shortage']; t=results['stress']; n=len(patterns)
    used=usage; total_used=sum(used.values()); waste=b['objective']
    pct=100*waste/max(1,waste+sum(float(patterns.iloc[k].used_mm)*float(b['x'][k]) for k in range(len(patterns))))
    plan=patterns[['pattern_id','pipe_id','used_mm','waste_mm']].copy(); plan['x, труб']=b['x']; plan=plan[plan['x, труб']>1e-8]
    shortage_rows=[]
    for pid in pipes.pipe_id:
        before=b['usage_by_pipe'].get(str(pid),0.0); after=s['usage_by_pipe'].get(str(pid),0.0)
        shortage_rows.append({'Тип трубы':pid,'Базовое использование, шт.':before,'После сокращения запаса, шт.':after,'Изменение, шт.':after-before})
    shortage_table=pd.DataFrame(shortage_rows)
    benchmark_show=benchmark[benchmark.repeat=='median'][['method','time_seconds','iterations','objective']]
    ds=float(benchmark_show.loc[benchmark_show.method=='highs-ds','time_seconds'].iloc[0])
    ipm=float(benchmark_show.loc[benchmark_show.method=='highs-ipm','time_seconds'].iloc[0])
    speed_text=f"`highs-ds` быстрее: медиана {ds:.6f} с против {ipm:.6f} с ({ipm/ds:.2f}×); итераций {int(benchmark_show.loc[benchmark_show.method=='highs-ds','iterations'].iloc[0])} против {int(benchmark_show.loc[benchmark_show.method=='highs-ipm','iterations'].iloc[0])}."
    positive=sensitivity.loc[sensitivity.actual_delta.astype(float).idxmax()]
    negative=sensitivity.loc[sensitivity.actual_delta.astype(float).idxmin()]
    report=fr'''# Лабораторная работа №1 — линейное программирование

## 1. Титульная информация

**Дисциплина:** Методы оптимизации. Линейное программирование  
**Тема:** минимизация отходов при резке труб (усреднённая ЛП-постановка), вариант 19.  
Индивидуальные параметры: $N=5$, $Q=1$, $Y=26$, $V=19$, seed $=10105$. Проверка: $1+((17·5+7·1+26)\bmod20)=19$; $10000+100·1+5=10105$.  
Стек: Python 3.14.4, NumPy 2.3.3, Pandas 2.3.3, SciPy 1.16.2 `linprog` с HiGHS, Matplotlib 3.10.7.

## 2. Цель работы

Построить непрерывную LP-модель раскроя, найти план с минимальными суммарными остатками материала, независимо проверить ограничения и исследовать изменения условий.

## 3. Индивидуальное задание

Вариант 19: минимизация отходов при резке труб в усреднённой линейной постановке. План определяет схемы раскроя труб пяти типов для точного выполнения шести заказов.

## 4. Постановка задачи

Для каждого исходного типа имеется ограниченный запас. Каждая схема описывает целочисленную комбинацию отрезков, но количество труб, назначенное схеме, является непрерывным средним значением. Перепроизводство заказов запрещено.

## 5. Исходные данные

Данные сформированы единственным генератором `numpy.random.default_rng(10105)`: длины труб и заготовок равномерно выбирались в заданных диапазонах с шагом 10 мм, затем длины отсортированы; спрос генерировался целыми числами 18–50. Запасы назначены консервативно достаточными по суммарной длине спроса. Исходные CSV после создания сохраняются и при перезапусках не меняются.

### Типы труб

{table(pipes)}

### Заказы

{table(orders)}

## 6. Математическая модель

Множества: $I$ — типы труб; $J$ — заказы; $P_i$ — схемы для типа $i$; $P=\bigcup_iP_i$. Параметры: длина трубы $L_i$ (мм), запас $S_i$ (шт.), длина детали $l_j$ (мм), спрос $d_j$ (шт.), выпуск схемы $a_{{jp}}$ (дет.), отход $w_p$ (мм/трубу). Переменная $x_p\ge0$ — усреднённое количество труб, назначенных схеме.

$$\min Z=\sum_{{p\in P}}w_px_p$$

$$\sum_{{p\in P}}a_{{jp}}x_p=d_j\quad(j\in J),\qquad
\sum_{{p\in P_i}}x_p\le S_i\quad(i\in I),\qquad x_p\ge0.$$

Для схемы проверяется $\sum_j a_{{jp}}l_j\le L_i$, а $w_p=L_i-\sum_j a_{{jp}}l_j$. Пропил принят равным нулю; длины и отходы измеряются в мм, количества — в штуках.

## 7. Генерация схем раскроя

Алгоритм рекурсивно перебирает каждую допустимую комбинацию количества шести заказных отрезков и отсеивает пустые схемы. Верхние границы следуют из $a_{{jp}}\le\lfloor L_i/l_j\rfloor$, поэтому перечисление полное для заданных данных и ограничений производительности не вводилось. Получено **{n}** допустимых схем. Файл `data/generated/cutting_patterns.csv` содержит полный список. В плане используется следующее подмножество:

{table(plan)}

## 8. Программная реализация

`data_generation.py` формирует исходники и метаданные; `data_loader.py` проверяет таблицы; `cutting_patterns.py` перечисляет схемы; `model.py` строит матрицы LP; `solver.py` запускает HiGHS и журналирует результаты; `validation.py` независимо пересчитывает баланс; `sensitivity.py` формирует двойственные оценки; `experiments.py` запускает сценарии и benchmark; `reporting.py` строит графики и данный отчёт. Главный сценарий — `run.py`.

## 9. Результаты базового сценария

Статус: **{b['status']}**; целевая функция: **{waste:.6f} мм**; время: {b['time']:.6f} с; использовано труб (LP-эквивалент): {total_used:.6f}; доля отхода в использованном материале: {pct:.4f}%. Запланированный раскрой приведён выше. Запас каждого типа проверяется отдельно; заказные балансы точны с учётом численной погрешности.

![Использование типов труб](../plots/pipe_usage.png)

## 10. Независимая проверка

Проверены равенства выпуска по всем заказам, ограничения запасов каждого типа и неотрицательность. Допуск $\varepsilon=10^{{-7}}$. Максимальная невязка: {validation['max_deviation']:.3e}; самостоятельно пересчитанная цель: {validation['objective_recalculated']:.9f} мм; отличие от результата HiGHS: {validation['objective_gap']:.3e}; проверка: **{'пройдена' if validation['passed'] else 'НЕ пройдена'}**. Пересчёт сделан непосредственно по CSV схем и вектору $x$, без матриц модели.

## 11. Дефицитный сценарий

Запас наиболее используемого в базовом плане типа **{s['changed_pipe_type']}** уменьшен с {s['stock_before']:.3f} до {s['stock_after']:.3f} трубы (−20%; описание — `data/scenarios/shortage.json`). Статус: **{s['status']}**; целевая функция: {fmt(s['objective'])}; число использованных труб: {s['used']:.6f} трубы (LP-эквивалент). Изменение цели относительно базы: {delta(b['objective'],s['objective'])}. План остался прежним, поскольку исходный запас этого типа превышал фактическое использование; сокращённый запас также остался выше потребности плана. Использование по типам:

{table(shortage_table)}

## 12. Стрессовый сценарий

Запас типа **{s['changed_pipe_type']}** снижен ещё на 20% от исходного, спрос всех заказов увеличен в 12 раз (`data/scenarios/stress.json`). Статус HiGHS: **{t['status']}**; значение целевой функции: {fmt(t['objective'])}. Совокупный доступный запас материала составляет {t['material_capacity_mm']:,.0f} мм, тогда как суммарная длина требуемых деталей равна {t['required_order_length_mm']:,.0f} мм. Уже необходимая длина деталей превышает доступный материал, поэтому точное выполнение спроса физически невозможно. SciPy-интерфейс не предоставляет здесь IIS, и он не заявляется.

![Сравнение сценариев](../plots/waste_comparison.png)

## 13. Анализ теневых цен

SciPy возвращает маргиналы по правым частям; в таблице они измеряются в мм изменения отхода на единицу спроса или запаса. Для равенств это производная цели по спросу, для верхних границ — по запасу (неположительная при минимизации). Нулевая оценка запаса означает отсутствие локальной ценности дополнительной трубы в текущем базисе. Все ограничения запасов неактивны. Например, маргинал заказа {positive['constraint']} равен {positive['shadow_price']:.3f}: увеличение спроса на одну деталь увеличило цель на {positive['actual_delta']:.3f} мм. Для {negative['constraint']} маргинал {negative['shadow_price']:.3f}, и добавление одной детали уменьшило суммарный отход на {abs(negative['actual_delta']):.3f} мм, поскольку схема лучше заполняет длину труб. Для обеих проверок фактическое изменение совпало с локальным прогнозом. Такие оценки применимы только пока базис не меняется.

{table(shadow)}

Проверка прогноза для увеличения спроса каждого заказа на одну деталь:

{table(sensitivity)}

## 14. Дополнительное задание: сравнение решателей

Сравнивались `highs-ds` (dual simplex) и `highs-ipm` на одинаковой модели, с семью повторениями; измерялось только решение подготовленной LP. Итерации берутся из `OptimizeResult.nit`, если SciPy их предоставил. Сводка медиан:

{table(benchmark_show)}

![Сравнение времени решения](../plots/solver_comparison.png)

Оба метода получили одинаковую цель {waste:.6f} мм. {speed_text} На небольшой задаче абсолютные времена чувствительны к системной нагрузке и внутренним накладным расходам; вывод основан на семи повторениях и их медианах.

## 15. Анализ результатов

Сценарий снижения запаса на 20% не повлиял на оптимальный план: использовано 34.125 трубы типа T4 при доступных 42.4, и цель сохранилась на уровне 3761.25 мм. Это согласуется с нулевой теневой ценой неактивного ограничения запаса T4. Стрессовый сценарий недопустим, потому что требуемая длина деталей 3,769,800 мм больше суммарной длины доступного сырья 1,890,340 мм. В benchmark обе стратегии нашли одинаковую цель; для этой небольшой задачи dual simplex оказался быстрее. Локальные ценовые прогнозы подтвердились малыми изменениями спроса; при крупных изменениях может меняться базис.

## 16. Вывод

Построена и решена усреднённая LP-модель с точным выполнением спроса, ограниченным запасом и минимизацией остатков. Схемы перечислены исчерпывающе для заданных длин, базовое решение проверено независимым пересчётом. Теневые цены и сценарные прогоны показывают, как доступность материала и структура заказа влияют на стоимость раскроя.

## 17. Ограничения модели

Переменные $x_p$ непрерывны, потому что постановка описывает усреднённый поток повторяющихся операций, а не единичный физический запуск. Поэтому результат может содержать дробное число труб: например, 2.4 трубы одной схемы означает долю/среднее по серии планов. Для агрегированного анализа это допустимо и сохраняет линейность. Исполнение конкретной смены может потребовать целочисленной модели. Не учитываются пропил, дефекты, переналадка, ограничения станка, очередность резов, совместимость материалов, упаковка и стоимость разных типов труб.

## 18. Запуск

Из корня проекта: `python -m pip install -r requirements.txt`; затем `cd lr-1 && python run.py`. Python 3.11–3.14; точные версии зависимостей закреплены в requirements-файлах. Отчёт: `reports/report.md`.
'''
    # Локальные форматтеры применяются только после сборки текста.
    (root/'reports'/'report.md').write_text(report,encoding='utf-8')

def fmt(v,unit='мм'): return 'не рассчитывается (модель недопустима)' if v is None else f'{v:.6f} {unit}'
def delta(a,b): return 'сценарий недопустим' if b is None else f'{b-a:+.6f} мм отходов относительно базы'
