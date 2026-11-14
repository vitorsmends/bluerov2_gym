"""Summarize the adapted Walker W1--W3 eight-path campaign."""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
import pandas as pd
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from .finetune_controller_comparison import ROOT

SEAS=['walker_w1','walker_w2','walker_w3']
CONTROLLERS=[('ppo','PPO','#59A14F'),('ft','FT','#E69F00'),('nmpc','NMPC','#E15759'),('smc','SMC','#B07AA1')]


def generate(root):
    out=root/'comparison';out.mkdir(exist_ok=True)
    rows=[];frames={};hashes={};reference=None
    configs={key:yaml.safe_load((root/key/'config.resolved.yaml').read_text()) for key in ['baseline','ft']}
    for key in ['experiment','environment','disturbance','trajectories']:
        if configs['baseline'][key]!=configs['ft'][key]:raise ValueError(f'Different comparison setting: {key}')
    for folder in ['baseline','ft']:
        if yaml.safe_load((root/folder/'manifest.yaml').read_text())['status']!='completed':raise ValueError(f'Incomplete {folder}')
    states=['x','y','z','roll','pitch','yaw','u','v','w','p','q','r']
    for sea in SEAS:
        reference=None
        for ctrl,label,color in CONTROLLERS:
            folder='ft' if ctrl=='ft' else 'baseline';prefix='ppo' if ctrl=='ft' else ctrl
            path=root/folder/f'data/{sea}/{prefix}_figure_eight_depth.csv'
            hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
            df=pd.read_csv(path).sort_values(['repetition_id','time']).reset_index(drop=True)
            if reference is not None:
                cols=['repetition_id','seed','time','Hs','Tp','gamma','wave_dir_x','wave_dir_y','scale','max_current','noise_std']+[c for c in df if c.endswith('_ref')]
                np.testing.assert_allclose(reference[cols],df[cols],rtol=0,atol=1e-10)
                np.testing.assert_allclose(reference.groupby('repetition_id').first()[states],df.groupby('repetition_id').first()[states],rtol=0,atol=1e-10)
            reference=df
            df['yaw_abs_deg']=np.degrees(np.abs(np.arctan2(np.sin(df.yaw_error),np.cos(df.yaw_error))))
            for rep,g in df.groupby('repetition_id'):
                if len(g)!=1000 or not np.allclose(g.time,np.arange(1000)*.1):raise ValueError(f'Incomplete horizon: {path}')
                rows.append(dict(sea=sea,controller=label,repetition=int(rep),seed=int(g.seed.iloc[0]),
                    rmse_m=np.sqrt(np.mean(g.tracking_error_m**2)),yaw_mae_deg=g.yaw_abs_deg.mean(),
                    max_position_error_m=g.tracking_error_m.max(),post10_rmse_m=np.sqrt(np.mean(g[g.time>=10].tracking_error_m**2))))
            if df.repetition_id.nunique()!=3:raise ValueError('Expected three paired seeds')
            frames[sea,ctrl]=df
    data=pd.DataFrame(rows);data.to_csv(out/'run_metrics.csv',index=False)
    summary=data.groupby(['sea','controller'])[['rmse_m','yaw_mae_deg','max_position_error_m','post10_rmse_m']].agg(['mean',lambda x:x.std(ddof=0)]).rename(columns={'<lambda_0>':'std'})
    summary.to_csv(out/'summary.csv')
    def save(fig,name):
        for ext in ['pdf','png']:fig.savefig(out/f'{name}.{ext}',dpi=250,bbox_inches='tight')
        plt.close(fig)
    with plt.rc_context({'font.family':'serif','font.size':12,'pdf.fonttype':42}):
        for sea in SEAS:
            fig=plt.figure(figsize=(8.5,4.9));grid=fig.add_gridspec(2,2,width_ratios=[1.8,1],hspace=.32,wspace=.34)
            spatial=fig.add_subplot(grid[:,0],projection='3d');position=fig.add_subplot(grid[0,1]);yaw=fig.add_subplot(grid[1,1])
            ref=frames[sea,'ppo'].query('repetition_id==0');handles=spatial.plot(ref.x_ref,ref.y_ref,ref.z_ref,'k--',lw=1.5,label='Reference')
            for ctrl,label,color in CONTROLLERS:
                df=frames[sea,ctrl];single=df.query('repetition_id==0');handles+=spatial.plot(single.x,single.y,single.z,color=color,lw=1.2,label=label)
                for ax,column in [(position,'tracking_error_m'),(yaw,'yaw_abs_deg')]:
                    stats=df.groupby('time')[column].agg(['mean',lambda x:x.std(ddof=0)]);m,sd=stats.iloc[:,0].to_numpy(),stats.iloc[:,1].to_numpy()
                    ax.plot(stats.index,m,color=color,lw=1.2);ax.fill_between(stats.index,np.maximum(0,m-sd),m+sd,color=color,alpha=.13,lw=0)
            spatial.set_xlabel('X [m]',labelpad=0);spatial.set_ylabel('Y [m]',labelpad=0);spatial.set_zlabel('Z [m]',labelpad=2)
            spatial.set_box_aspect((1,1,.8));spatial.view_init(22,-60);spatial.invert_zaxis();spatial.tick_params(labelsize=9,pad=0)
            position.set_ylabel('Position error [m]');yaw.set_ylabel('Yaw error [deg]');yaw.set_xlabel('Time [s]')
            for ax in [position,yaw]:ax.grid(alpha=.2);ax.set_xlim(0,100)
            position.tick_params(labelbottom=False)
            fig.legend(handles=handles,loc='upper center',ncol=5,frameon=False,fontsize=11)
            fig.subplots_adjust(left=.03,right=.97,top=.84,bottom=.13);save(fig,sea)
        fig,axes=plt.subplots(1,2,figsize=(8.5,3.7))
        for ax,metric,ylabel in zip(axes,['rmse_m','yaw_mae_deg'],['Position RMSE [m]','Yaw MAE [deg]']):
            for i,sea in enumerate(SEAS):
                values=[summary.loc[(sea,label),(metric,'mean')] for _,label,_ in CONTROLLERS]
                errors=[summary.loc[(sea,label),(metric,'std')] for _,label,_ in CONTROLLERS]
                bars=ax.bar(np.arange(4)+(i-1)*.24,values,width=.22,yerr=errors,capsize=2,color=['#b9d6ea','#629ac2','#245b83'][i],label=sea.replace('walker_','').upper())
                for bar in bars:bar.set_hatch(['','///','xx'][i])
            ax.set_xticks(np.arange(4),[label for _,label,_ in CONTROLLERS]);ax.set_ylabel(ylabel);ax.grid(axis='y',alpha=.2)
        fig.legend(*axes[0].get_legend_handles_labels(),loc='upper center',ncol=3,frameon=False)
        fig.tight_layout(rect=(0,0,1,.90));save(fig,'metrics_mean_std')
    for cfg in configs.values():
        for controller in cfg['controllers']:
            if controller['name']=='ppo':
                for key in ['model_path','vecnormalize_path']:
                    path=Path(controller['params'][key]);hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    (out/'provenance.json').write_text(json.dumps(dict(sources_sha256=hashes,bands='Population SD over three paired wave seeds; one trained PPO and one FT',spatial_seed=42,reference='W1-W3 Hs/Tp and depths from Walker et al. IJRR DOI 10.1177/02783649241286909',differences='Analytic first-order JONSWAP gamma=3.3 rather than exact buoy spectra; six-DOF current mechanical and empirical load model; eight instead of station keeping; blind NMPC; 100s episodes'),indent=2))
    print(summary[['rmse_m','yaw_mae_deg']].round(4).to_string())

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,default=ROOT/'artifacts/evaluation/walker_eight');a=p.parse_args();generate(a.run)
