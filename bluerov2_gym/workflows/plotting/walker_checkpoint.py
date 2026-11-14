"""Compare a frozen intermediate Walker PPO checkpoint with archived controllers."""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
import pandas as pd
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .finetune_controller_comparison import ROOT
from .walker_eight import SEAS


def generate(run):
    meta=yaml.safe_load((run/'checkpoint_provenance.yaml').read_text())
    previous=ROOT/'artifacts/evaluation/walker_eight'
    variants=[('PPO novo (B)',run/'evaluation','ppo','#0072B2'),('PPO artigo',previous/'baseline','ppo','#59A14F'),
              ('FT',previous/'ft','ppo','#E69F00'),('NMPC',previous/'baseline','nmpc','#E15759'),('SMC',previous/'baseline','smc','#B07AA1')]
    current=yaml.safe_load((run/'evaluation/config.resolved.yaml').read_text())
    for folder in {v[1] for v in variants}:
        cfg=yaml.safe_load((folder/'config.resolved.yaml').read_text())
        for key in ['experiment','environment','disturbance','trajectories']:
            if cfg[key]!=current[key]:raise ValueError(f'Different {key}: {folder}')
        if yaml.safe_load((folder/'manifest.yaml').read_text())['status']!='completed':raise ValueError(f'Incomplete run: {folder}')
    out=run/'comparison';out.mkdir(exist_ok=True);frames={};rows=[];hashes={}
    for sea in SEAS:
        reference=None
        for name,folder,prefix,color in variants:
            path=folder/f'data/{sea}/{prefix}_figure_eight_depth.csv';hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
            df=pd.read_csv(path).sort_values(['repetition_id','time']).reset_index(drop=True)
            if reference is not None:
                columns=['seed','repetition_id','time','Hs','Tp','gamma','wave_dir_x','wave_dir_y','scale','max_current','noise_std']+[c for c in df if c.endswith('_ref')]
                np.testing.assert_allclose(reference[columns],df[columns],rtol=0,atol=1e-10)
                state=['x','y','z','roll','pitch','yaw','u','v','w','p','q','r']
                np.testing.assert_allclose(reference.groupby('repetition_id').first()[state],df.groupby('repetition_id').first()[state],rtol=0,atol=1e-10)
            reference=df
            df['yaw_abs_deg']=np.degrees(np.abs(np.arctan2(np.sin(df.yaw_error),np.cos(df.yaw_error))))
            if df.repetition_id.nunique()!=3:raise ValueError('Expected three paired repetitions')
            for rep,g in df.groupby('repetition_id'):
                if len(g)!=1000 or not np.allclose(g.time,np.arange(1000)*.1):raise ValueError(f'Incomplete horizon {path}')
                rows.append(dict(sea=sea,controller=name,repetition=int(rep),seed=int(g.seed.iloc[0]),rmse_m=np.sqrt(np.mean(g.tracking_error_m**2)),yaw_mae_deg=g.yaw_abs_deg.mean()))
            frames[sea,name]=df
    values=pd.DataFrame(rows);values.to_csv(out/'run_metrics.csv',index=False)
    summary=values.groupby(['sea','controller'])[['rmse_m','yaw_mae_deg']].agg(['mean',lambda v:v.std(ddof=0)]).rename(columns={'<lambda_0>':'std'})
    summary.to_csv(out/'summary.csv')
    with plt.rc_context({'font.family':'serif','font.size':12,'pdf.fonttype':42}):
        for sea in SEAS:
            fig=plt.figure(figsize=(9,5.2));grid=fig.add_gridspec(2,2,width_ratios=[1.8,1],wspace=.32,hspace=.28)
            spatial=fig.add_subplot(grid[:,0],projection='3d');position=fig.add_subplot(grid[0,1]);yaw=fig.add_subplot(grid[1,1])
            ref=frames[sea,variants[0][0]].query('repetition_id==0');handles=spatial.plot(ref.x_ref,ref.y_ref,ref.z_ref,'k--',lw=1.5,label='Reference')
            for name,_,_,color in variants:
                df=frames[sea,name];single=df.query('repetition_id==0');handles+=spatial.plot(single.x,single.y,single.z,color=color,lw=1.2,label=name)
                for ax,column in [(position,'tracking_error_m'),(yaw,'yaw_abs_deg')]:
                    stats=df.groupby('time')[column].agg(['mean',lambda v:v.std(ddof=0)]);mean,sd=stats.iloc[:,0].to_numpy(),stats.iloc[:,1].to_numpy()
                    ax.plot(stats.index,mean,color=color,lw=1.2);ax.fill_between(stats.index,np.maximum(0,mean-sd),mean+sd,color=color,alpha=.12,lw=0)
            spatial.set_xlabel('X [m]',labelpad=0);spatial.set_ylabel('Y [m]',labelpad=0);spatial.set_zlabel('Z [m]',labelpad=1)
            spatial.view_init(22,-60);spatial.invert_zaxis();spatial.set_box_aspect((1,1,.8));spatial.tick_params(labelsize=9,pad=0)
            position.set_ylabel('Position error [m]');yaw.set_ylabel('Yaw error [deg]');yaw.set_xlabel('Time [s]')
            for ax in [position,yaw]:ax.grid(alpha=.2);ax.set_xlim(0,100)
            position.tick_params(labelbottom=False)
            fig.legend(handles=handles,loc='upper center',ncol=3,frameon=False,fontsize=11)
            fig.subplots_adjust(left=.03,right=.97,top=.78,bottom=.13)
            for ext in ['pdf','png']:fig.savefig(out/f'{sea}.{ext}',dpi=250,bbox_inches='tight')
            plt.close(fig)
    (out/'provenance.json').write_text(json.dumps(dict(checkpoint=meta,source_csv_sha256=hashes,spatial_seed=42,bands='Population SD across three paired environmental seeds; not across policy training seeds',reuse='Only new PPO simulated; archived PPO/FT/NMPC/SMC CSVs unchanged'),indent=2))
    (out/'README.md').write_text(f'''# Intermediate Walker PPO comparison

Checkpoint: {meta['checkpoint_step']} PPO steps, phase {meta['phase']} (std W1 preparation). Training is NOT complete and continues independently. Matching model and normalizer copied under ../inputs/ before evaluation. Stage-assessment RMSE at snapshot: {meta['assessment_rmse_m']:.6f} m; success fraction {meta['assessment_success_rate']}.

Only the new PPO was simulated: W1--W3, eight at -5 m, paired seeds 42--44, 100 s. Classical controllers and original PPO/FT are reused from ../../walker_eight after checking simulator, experiment, reference and initial-state equality. Evaluation uses TARGET mechanics, whereas this intermediate checkpoint is still training in STD mechanics; the results cannot establish final-curriculum performance.

Spatial panels show seed 42. Error curves show mean +/- population SD of three realizations. Tables average run-level RMSE and yaw MAE. The figure label PPO novo (B) identifies this intermediate checkpoint. The training process and its source checkpoints were not modified.
''')
    print(summary.round(4).to_string())

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);a=p.parse_args();generate(a.run)
