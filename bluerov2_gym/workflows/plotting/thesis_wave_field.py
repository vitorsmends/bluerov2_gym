"""Legible thesis wave-elevation map and cross-section from the benchmark generator."""
import hashlib,json
import numpy as np
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from .finetune_controller_comparison import ROOT
from bluerov2_gym.envs.core.dynamics import Dynamics


def generate():
    source=ROOT/'artifacts/evaluation/physics_wave_curriculum_yaw_sea_states/config.resolved.yaml'
    cfg=yaml.safe_load(source.read_text());params=dict(cfg['disturbance']['scenarios']['moderate'],seed=42)
    model=Dynamics(jonswap_params=params,dynamics_config=cfg['environment']['dynamics_config'])
    def elevation(x,y):
        x,y=np.broadcast_arrays(x,y);eta=np.zeros_like(x,dtype=float)
        for w in model._waves:
            phase=w['k']*(x*np.cos(w['theta'])+y*np.sin(w['theta']))-w['omega']*55+w['phase']
            eta+=w['A']*np.cos(phase)
        return eta
    out=ROOT/'documents/msc_thesis/images/path_tracking/chapter2_wave';out.mkdir(exist_ok=True,parents=True)
    x=np.linspace(-25,25,180);X,Y=np.meshgrid(x,x);Z=elevation(X,Y);section=elevation(x,0.)
    with plt.rc_context({'font.family':'serif','font.size':12,'pdf.fonttype':42}):
        fig,axes=plt.subplots(1,2,figsize=(7.2,3.4),layout='constrained',gridspec_kw={'width_ratios':[1.15,1]})
        limit=np.max(np.abs(Z));norm=TwoSlopeNorm(vmin=-limit,vcenter=0,vmax=limit)
        image=axes[0].pcolormesh(X,Y,Z,cmap='RdBu_r',norm=norm,shading='auto',rasterized=True)
        axes[0].axhline(0,color='black',ls='--',lw=1)
        axes[0].plot(0,0,'ko',ms=3)
        axes[0].set(xlabel='X [m]',ylabel='Y [m]',aspect='equal',xticks=[-20,0,20],yticks=[-20,0,20])
        axes[0].set_title('(a) Elevation map',fontsize=12)
        colorbar=fig.colorbar(image,ax=axes[0],shrink=.85,pad=.02)
        colorbar.ax.set_title('[m]',fontsize=10,pad=6)
        axes[1].plot(x,section,color='#1678B5',lw=1.5,label=r'$Y=0$ m')
        axes[1].axhline(0,color='.35',ls='--',lw=1,label='Still water')
        axes[1].set(xlabel='X [m]',ylabel=r'$\zeta$ [m]',xlim=(-25,25),ylim=(-limit,limit))
        axes[1].set_title('(b) Spatial cross-section',fontsize=12);axes[1].grid(alpha=.2)
        axes[1].legend(fontsize=10,loc='upper right')
        for ext in ['pdf','png']:fig.savefig(out/f'directional_field_compact.{ext}',dpi=300,bbox_inches='tight')
        plt.close(fig)
    (out/'directional_field_compact.json').write_text(json.dumps({'config':str(source),'sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'scenario':'moderate','seed':42,'snapshot_s':55,'slice_y_m':0,'note':'Same wave components; visualization only. No vehicle simulation or policy evaluation.'},indent=2))
    print(out/'directional_field_compact.pdf')

if __name__=='__main__':generate()
