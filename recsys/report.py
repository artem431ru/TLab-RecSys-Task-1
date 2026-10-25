"""Create an honest short report from actual experiment JSON outputs."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def read(p):
    with open(p,encoding='utf-8') as f:return json.load(f)


def fmt(x):
    return '—' if x is None else f'{x:.4f}' if isinstance(x,float) else str(x)


def report(task1,task2,out):
    a=read(task1);b=read(task2)
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    demos=any(r.get('source')=='synthetic_test' for r in (a,b))
    warning=('**ВНИМАНИЕ: это результаты на СИНТЕТИЧЕСКИХ ДАННЫХ, а не выводы по KION!**\n\n'
             if demos else '')
    task1exp=a['experiments']
    pre=b['reference']['test'];post=b['after_rl']['test']
    lines=[f'# Два исследования рекомендательных систем\n\n{warning}',
         '## Эксперимент 1: автоматизация Feature Engineering\n',
         'Датасет: '+a['source']+'. Временные интервалы: '+', '.join(a['boundaries'])+'.',
         'Признаки строго сформированы из событий до соответствующего момента предсказания.\n',
         '| Модель | Число признаков | NDCG@10 | Recall@10 |',
         '|---|---:|---:|---:|']
    for name,x in task1exp.items():
        lines.append(f"| {name} | {x['features']} | {fmt(x['ndcg@10'])} | {fmt(x['recall@10'])} |")
    lines.extend(['\nВажное ограничение: кандидаты выбираются из исторического top-N популярных объектов; '
                  'рассматриваются только пользователи, у которых есть хотя бы один наблюдаемый положительный объект в пуле. '
                  'Сравнение корректно **в рамках этого офлайн-протокола**, но не измеряет промышленное качество рекомендаций.',
                  '\n## Эксперимент 2: GPT-2 + модель награды + REINFORCE\n',
                  f"Вариант архитектуры: {b['backend']}; размер словаря: {b['vocab']}; устройство: {b['device']}.",
                  '| Метрика на тесте | До RL | После RL |', '|---|---:|---:|'])
    for key in ['recall@10','ndcg@10','predicted_reward_topk_mean','distinct_recommendations','mean_normalized_item_popularity_topk']:
        lines.append(f'| {key} | {fmt(pre[key])} | {fmt(post[key])} |')
    lines.extend([f"\nMSE reward model на наблюдаемых test-откликах: {fmt(pre['observed_reward_mse'])}.",
                  '**Важно:** награда на рекомендованных (непросмотренных) объектах оценивается только Reward Model. '
                  'Рост её предсказания НЕ подтверждает рост реальной удовлетворённости. '
                  'Для контрфактических выводов потребуются рандомизированные показы либо корректный OPE-протокол.',
                  '\n## Что исследовать глубже\n',
                  '1. Зависимость качества от ширины окна и количества признаков; отбор по валидации.',
                  '2. Две величины KL-штрафа при RL и влияние на точность и популярность рекомендаций.',
                  '3. Смещение reward model, короткие пользовательские истории и невозможность наблюдать отклик на невыданный объект.',
                  '\n## Воспроизводимость\n',
                  'Команды для запуска приведены в корневом README.md; код не использует будущие события в построении признаков.'])
    (out/'REPORT_RU.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    labels=['manual','automatic_grid','automatic_selected']
    fig,ax=plt.subplots(figsize=(8,4))
    ax.bar(labels,[task1exp[n]['ndcg@10'] for n in labels]);ax.set_ylabel('NDCG@10');ax.set_title('Experiment 1: test comparison')
    fig.tight_layout();fig.savefig(out/'task1_ndcg.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4))
    ax.bar(['GPT-2','GPT-2 + RL'],[pre['ndcg@10'],post['ndcg@10']]);ax.set_ylabel('NDCG@10');ax.set_title('Experiment 2: test comparison')
    fig.tight_layout();fig.savefig(out/'task2_ndcg.png',dpi=150);plt.close(fig)
    return out/'REPORT_RU.md'

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--task1',default='results/task1/task1_metrics.json')
    p.add_argument('--task2',default='results/task2/task2_metrics.json')
    p.add_argument('--out',default='results')
    args=p.parse_args()
    print(report(args.task1,args.task2,args.out))
