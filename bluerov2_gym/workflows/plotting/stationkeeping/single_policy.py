"""Original one-setpoint diagnostic drawing, reusable from saved CSVs."""
import matplotlib.pyplot as plt
import numpy as np

def plot_errors(data, target, wave_preset, terminated, truncated, output):
    error = data[:,13:16]
    norm = data[:,16]
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(3,1,figsize=(11,9),sharex=True,layout='constrained')
    t=data[:,0]
    for i,label in enumerate(['x','y','z']):
        axes[0].plot(t,error[:,i],label=f'$e_{label}$')
    axes[0].axhline(0,color='gray',lw=.7)
    axes[0].set_ylabel('Erro de posição (m)');axes[0].legend(ncol=3)
    axes[1].plot(t,norm,color='#176b87',label='Erro euclidiano')
    axes[1].axhline(.2,color='gray',ls='--',label='Referência: 0,20 m')
    axes[1].set_ylabel('Norma do erro (m)');axes[1].legend()
    for i,label in enumerate(['roll','pitch','yaw']):
        axes[2].plot(t,np.rad2deg(data[:,17+i]),label=label)
    axes[2].axhline(0,color='gray',lw=.7)
    axes[2].set_ylabel('Erro angular (graus)');axes[2].set_xlabel('Tempo (s)');axes[2].legend(ncol=3)
    for ax in axes:ax.grid(alpha=.2)
    status='encerrado antecipadamente' if terminated or truncated else 'concluído'
    fig.suptitle(f'PPO existente — station keeping | setpoint {tuple(float(v) for v in target)} m\n'
                 f'Mecânica corrigida | ondas {wave_preset} | {status}')
    for extension in ['png','pdf']:
        fig.savefig(output/f'errors.{extension}',dpi=180)
    plt.close(fig)

