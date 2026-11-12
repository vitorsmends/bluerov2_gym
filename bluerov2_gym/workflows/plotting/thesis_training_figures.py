"""Thesis-sized training and pooled-error figures, without changing recorded data."""
import numpy as np
import pandas as pd
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from .finetune_controller_comparison import ROOT
from .training.thesis import CurriculumFigures

LABELS={'Reward bruto médio por passo':'Mean raw reward\nper step',
'Erro euclidiano médio (m)':'Mean position\nerror [m]',
'RMSE por rollout (m)':'Position RMSE\nper rollout [m]',
'Passos de treinamento no ambiente':'PPO environment steps',
'Episódios aprovados (%)':'Successful\nepisodes [%]',
'Média do RMSE por episódio (m)':'Mean episode\nposition RMSE [m]',
'Média do RMSE dos episódios (m)':'Mean episode\nposition RMSE [m]'}

class ThesisTrainingFigures(CurriculumFigures):
    def save(self,fig,name):
        fig.set_layout_engine(None)
        fig.set_size_inches(6.5,5.6 if name=='curriculum_learning' else 4.6)
        for ax in fig.axes:
            if ax.get_legend():ax.get_legend().remove()
            ax.set_ylabel(LABELS.get(ax.get_ylabel(),ax.get_ylabel()),fontsize=14,labelpad=8)
            ax.set_xlabel(LABELS.get(ax.get_xlabel(),ax.get_xlabel()),fontsize=14)
            ax.tick_params(labelsize=12)
        handles=[Patch(color=self.colors[i],alpha=.5,label=p['name'].split(' - ')[0]) for i,p in enumerate(self.phases)]
        fig.legend(handles=handles,loc='upper center',ncol=6,frameon=False,fontsize=12,columnspacing=.8,handlelength=1)
        fig.subplots_adjust(left=.24,right=.97,bottom=.13,top=.88,hspace=.28)
        super().save(fig,name)


def generate():
    train=ROOT/'artifacts/training/physics_wave_curriculum_yaw'
    out=ROOT/'documents/msc_thesis/images/path_tracking/current/compact_training'
    settings=dict(log_dir=str(train/'logs/tensorboard'),output_dir=str(out),table_dir=str(ROOT/'artifacts/tables/thesis_training'),smooth_window=10,
      curriculum_status=yaml.safe_load((train/'models/curriculum_status.yaml').read_text()),training_config=yaml.safe_load((train/'config.resolved.yaml').read_text()))
    with plt.rc_context({'font.family':'serif','font.size':14,'pdf.fonttype':42}):
        figures=ThesisTrainingFigures(settings);figures.learning();figures.validation()
        data=ROOT/'artifacts/evaluation/star_finetune_validation/article_all_controllers/data'
        colors={'PID':'#4E79A7','SMC':'#B07AA1','NMPC':'#E15759','PPO':'#59A14F','FT':'#E69F00'}
        for metric,label,name in [('tracking_error_m','Position error [m]','path_tracking_error_mean_std'),('yaw_error','Absolute yaw error [deg]','yaw_error_mean_std')]:
            fig,ax=plt.subplots(figsize=(6.0,3.5))
            for ctrl,color in colors.items():
                runs=[]
                for path in sorted(data.glob(f'*/{ctrl.lower()}_*.csv')):
                    df=pd.read_csv(path)
                    for _,g in df.groupby('repetition_id'):
                        values=g.sort_values('time')[metric].to_numpy()
                        if metric=='yaw_error':values=np.degrees(np.abs(np.arctan2(np.sin(values),np.cos(values))))
                        runs.append(values)
                if len(runs)!=27:raise ValueError(f'Expected 27 runs for {ctrl}')
                values=np.array(runs);mean,sd=values.mean(axis=0),values.std(axis=0);time=np.arange(1000)*.1
                ax.plot(time,mean,color=color,label=ctrl,lw=1.4)
                # Signed lower position band is retained, matching the article.
                lower=mean-sd if metric=='tracking_error_m' else np.maximum(0,mean-sd)
                ax.fill_between(time,lower,mean+sd,color=color,alpha=.15,lw=0)
            ax.set_xlabel('Time [s]');ax.set_ylabel(label);ax.set_xlim(0,100);ax.grid(alpha=.25)
            fig.legend(*ax.get_legend_handles_labels(),loc='upper center',ncol=5,frameon=False,fontsize=12)
            fig.tight_layout(rect=(0,0,1,.88))
            for ext in ['pdf','png']:fig.savefig(out/f'{name}.{ext}',dpi=300,bbox_inches='tight')
            plt.close(fig)
    print(out)

if __name__=='__main__':generate()
