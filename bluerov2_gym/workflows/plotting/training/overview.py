"""Logged training performance with explicit handling of missing physical metrics."""
import csv
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import yaml
from .curves import find_run, moving_average


def main(settings=None):
    settings=settings or {}
    run,metrics=find_run(Path(settings['log_dir']),settings.get('run'))
    output=Path(settings['output_dir']);output.mkdir(parents=True,exist_ok=True)
    tables=Path(settings.get('table_dir',output/'tables'));tables.mkdir(parents=True,exist_ok=True)
    fig,axes=plt.subplots(3,1,figsize=(11,9),sharex=True,layout='constrained')
    reward_tag = ('rollout/ep_rew_mean' if 'rollout/ep_rew_mean' in metrics
                  else 'rollout/reward_mean_per_step')
    reward_label = ('Reward médio por episódio' if reward_tag == 'rollout/ep_rew_mean'
                    else 'Reward bruto médio por passo')
    specifications=[(reward_tag,reward_label,'#59A14F'),
                    ('rollout/position_error_mean_m','Erro euclidiano médio (m)','#4E79A7'),
                    ('rollout/ep_len_mean','Duração média do episódio (passos)','#B07AA1')]
    exported=[];summary={}
    window=int(settings.get("smooth_window",10))
    for ax,(tag,label,color) in zip(axes,specifications):
        ax.set_ylabel(label);ax.grid(alpha=.2)
        if tag not in metrics:
            ax.text(.5,.5,'Métrica ainda não registrada neste treino.',ha='center',va='center',transform=ax.transAxes)
            summary[tag]={'available':False};continue
        steps,values=metrics[tag]
        smoothed=moving_average(values,window)
        ax.plot(steps,values,color=color,alpha=.3,label='Valor registrado')
        ax.plot(steps[-len(smoothed):],smoothed,color=color,label=f'Média móvel ({min(window,len(values))} registros)')
        ax.legend()
        summary[tag]=dict(available=True,first=float(values[0]),last=float(values[-1]),records=len(values))
        exported.extend([int(step),tag,float(value)] for step,value in zip(steps,values))
    config=settings.get('training_config',{})
    rollout=int(config.get('ppo',{}).get('n_steps',4096))
    boundaries=[]
    if config.get('curriculum_transition',{}).get('mode')=='success':
        boundaries=[item['end_step'] for item in settings.get('curriculum_status',{}).get('history',[])
                    if item.get('status')=='mastered' and item['index']<len(config.get('curriculum',[]))]
    else:
        boundary=0
        for phase in config.get('curriculum',[])[:-1]:
            boundary+=int(np.ceil(phase['timesteps']/rollout))*rollout
            boundaries.append(boundary)
    for boundary in boundaries:
        for ax in axes:ax.axvline(boundary,color='gray',ls=':',lw=.8)
    axes[-1].set_xlabel('Passos de interação com o ambiente (não épocas internas do PPO)')
    fig.suptitle('Evolução registrada durante o treinamento\nReward conforme unidade indicada no eixo; pontilhados: transições do currículo')
    for extension in ['png','pdf']:fig.savefig(output/f'training_progress.{extension}',dpi=180)
    plt.close(fig)
    with (tables/'training_scalars.csv').open('w',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['environment_steps','metric','value']);writer.writerows(exported)
    (tables/'training_progress.yaml').write_text(yaml.safe_dump(summary,sort_keys=False))
