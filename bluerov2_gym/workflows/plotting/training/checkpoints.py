"""Plot retrospective checkpoint measurements, never masquerading as training logs."""
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd


def main(settings=None):
    settings=settings or {}
    source=Path(settings['input_dir'])/'checkpoint_metrics.csv'
    data=pd.read_csv(source).sort_values('training_steps')
    output=Path(settings['output_dir']);output.mkdir(parents=True,exist_ok=True)
    x=data['training_steps']
    fig,axes=plt.subplots(3,1,figsize=(10,9),sharex=True,layout='constrained')
    axes[0].plot(x,data['position_error_mean_m'],'o-',label='Erro euclidiano médio')
    axes[0].plot(x,data['position_error_rmse_m'],'s-',label='RMSE da posição')
    axes[0].axhline(.2,color='gray',ls='--',label='0,20 m')
    failures=data['terminated'].astype(bool)|data['truncated'].astype(bool)
    if failures.any():axes[0].scatter(x[failures],data.loc[failures,'position_error_rmse_m'],marker='x',s=90,color='red',label='Encerramento antecipado',zorder=5)
    axes[0].set_ylabel('Erro de posição (m)');axes[0].legend()
    axes[1].plot(x,data['reward_mean_per_step'],'o-',color='#59A14F')
    axes[1].set_ylabel('Reward médio por passo')
    axes[2].plot(x,100*data['success_fraction'],'o-',color='#B07AA1')
    axes[2].set_ylabel('Passos com erro < 0,20 m (%)');axes[2].set_ylim(-2,102)
    axes[2].set_xlabel('Passos de treinamento do checkpoint')
    for ax in axes:ax.grid(alpha=.25)
    fig.suptitle('Reavaliação retrospectiva dos checkpoints\nMesmo cenário e mesma seed; não é o erro registrado durante o treino')
    for extension in ['png','pdf']:fig.savefig(output/f'checkpoint_learning_curve.{extension}',dpi=180)
    plt.close(fig)
