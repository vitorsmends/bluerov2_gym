"""Compact thesis spatial views from paired current-policy evaluation CSVs."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from .finetune_controller_comparison import ROOT, POLICIES


def generate():
    campaign=ROOT/'artifacts/evaluation/star_finetune_validation'
    output=ROOT/'documents/msc_thesis/images/path_tracking/current/compact_trajectories'
    output.mkdir(parents=True,exist_ok=True)
    cases=['reference_889_figure_eight','square','circle'];names=['Figure-eight','Square','Circle']
    frames={};runs={};hashes={}
    for sea in ['calm','moderate','rough']:
        for case in cases+(['star'] if sea=='moderate' else []):
            original=ROOT/'artifacts/evaluation'/('physics_wave_curriculum_yaw_star' if case=='star' else 'physics_wave_curriculum_yaw_sea_states')
            tuned=campaign/('combined_star' if case=='star' else 'combined_benchmark')
            ref=None
            for policy,_,_ in POLICIES:
                folder=tuned if policy=='ft_star' else original
                ctrl='nmpc' if policy=='nmpc' else 'ppo'
                path=folder/f'data/{sea}/{ctrl}_{case}.csv'
                hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
                runs[sea,case,policy]=pd.read_csv(path)
                frame=runs[sea,case,policy].query('repetition_id == 0').sort_values('time')
                if len(frame)!=1000 or not np.allclose(frame.time,np.arange(1000)*.1):raise ValueError(f'Incomplete data: {path}')
                cols=['seed','time','Hs','Tp','gamma','wave_dir_x','wave_dir_y','scale','max_current','noise_std']+[c for c in frame if c.endswith('_ref')]
                if ref is not None:np.testing.assert_allclose(ref[cols],frame[cols],rtol=0,atol=1e-10)
                ref=frame;frames[sea,case,policy]=frame
    def save(fig,name):
        for ext in ['pdf','png']:fig.savefig(output/f'{name}.{ext}',dpi=250,bbox_inches='tight',pad_inches=.08)
        plt.close(fig)
    with plt.rc_context({'font.family':'serif','font.size':12,'pdf.fonttype':42}):
        for sea in ['calm','moderate','rough']:
            fig=plt.figure(figsize=(9,3.6))
            for col,case in enumerate(cases):
                ax=fig.add_subplot(1,3,col+1,projection='3d')
                ref=frames[sea,case,'ppo_article'];lines=ax.plot(ref.x_ref,ref.y_ref,ref.z_ref,'k--',lw=1.5,label='Reference')
                for policy,label,color in POLICIES:
                    df=frames[sea,case,policy];lines+=ax.plot(df.x,df.y,df.z,color=color,lw=1.2,label=label)
                ax.set_title(names[col],fontsize=13,pad=6)
                ax.set_xlim(-1.15,1.15);ax.set_ylim(-1.15,1.15);ax.set_zlim(-.75,.08);ax.invert_zaxis()
                ax.view_init(22,-60);ax.set_proj_type('ortho');ax.set_box_aspect((1,1,.8))
                for axis in [ax.xaxis,ax.yaxis,ax.zaxis]:axis.set_major_locator(MaxNLocator(3))
                ax.tick_params(labelsize=10,pad=0)
                ax.set_xlabel('X [m]',labelpad=2);ax.set_ylabel('Y [m]',labelpad=2)
                if col==2:fig.text(.985,.44,'Z [m]',rotation=90,va='center',fontsize=12)
            fig.legend(handles=lines,loc='upper center',ncol=4,frameon=False,fontsize=13)
            fig.subplots_adjust(left=.01,right=.94,top=.78,bottom=.14,wspace=.20)
            save(fig,f'trajectories_{sea}')
        for case,name in [('reference_889_figure_eight','eight'),('star','star')]:
            fig=plt.figure(figsize=(8,4.8));grid=fig.add_gridspec(2,2,width_ratios=[1.9,1],height_ratios=[1,1],hspace=.32,wspace=.36)
            ax=fig.add_subplot(grid[:,0],projection='3d');depth=fig.add_subplot(grid[0,1]);yawax=fig.add_subplot(grid[1,1])
            ref=frames['moderate',case,'ppo_article'];lines=ax.plot(ref.x_ref,ref.y_ref,ref.z_ref,'k--',lw=1.5,label='Reference')
            depth.plot(ref.time,ref.z_ref,'k--',lw=1.5)
            for policy,label,color in POLICIES:
                df=frames['moderate',case,policy];lines+=ax.plot(df.x,df.y,df.z,color=color,lw=1.2,label=label)
                depth.plot(df.time,df.z,color=color,lw=1.2)
                all_runs=runs['moderate',case,policy].copy()
                all_runs['yaw_deg']=np.degrees(np.abs(np.arctan2(np.sin(all_runs.yaw_error),np.cos(all_runs.yaw_error))))
                stats=all_runs.groupby('time').yaw_deg.agg(['mean',lambda v:v.std(ddof=0)])
                mean,sd=stats.iloc[:,0].to_numpy(),stats.iloc[:,1].to_numpy()
                yawax.plot(stats.index,mean,color=color,lw=1.2)
                yawax.fill_between(stats.index,np.maximum(0,mean-sd),mean+sd,color=color,alpha=.15,lw=0)
            ax.set_xlabel('X [m]',labelpad=0);ax.set_ylabel('Y [m]',labelpad=0);ax.set_zlabel('Z [m]',labelpad=1)
            ax.invert_zaxis();ax.view_init(22,-60);ax.set_box_aspect((1,1,.8));ax.tick_params(labelsize=9,pad=0)
            for axis in [ax.xaxis,ax.yaxis,ax.zaxis]:axis.set_major_locator(MaxNLocator(3))
            for chart in [depth,yawax]:chart.set_xlim(0,100);chart.set_xlabel('Time [s]');chart.grid(alpha=.2)
            depth.set_ylabel('Z [m]');yawax.set_ylabel('Yaw error [deg]')
            depth.set_xlabel('');depth.tick_params(labelbottom=False)
            yawax.xaxis.set_major_locator(MaxNLocator(3))
            depth.xaxis.set_major_locator(MaxNLocator(3))
            fig.legend(handles=lines,loc='upper center',ncol=4,frameon=False,fontsize=12)
            fig.subplots_adjust(top=.83,bottom=.14,left=.025,right=.97)
            save(fig,f'{name}_moderate')
        fig,axes=plt.subplots(1,2,figsize=(8,3.2))
        for ax,case,title in zip(axes,['square','circle'],['Square','Circle']):
            for policy,label,color in POLICIES:
                df=runs['moderate',case,policy].copy()
                df['yaw_deg']=np.degrees(np.abs(np.arctan2(np.sin(df.yaw_error),np.cos(df.yaw_error))))
                stats=df.groupby('time').yaw_deg.agg(['mean',lambda v:v.std(ddof=0)])
                mean,sd=stats.iloc[:,0].to_numpy(),stats.iloc[:,1].to_numpy()
                ax.plot(stats.index,mean,color=color,label=label,lw=1.2)
                ax.fill_between(stats.index,np.maximum(0,mean-sd),mean+sd,color=color,alpha=.15,lw=0)
            ax.set_title(title);ax.set_xlabel('Time [s]');ax.set_xlim(0,100);ax.grid(alpha=.2)
        axes[0].set_ylabel('Absolute yaw error [deg]')
        fig.legend(*axes[0].get_legend_handles_labels(),loc='upper center',ncol=3,frameon=False)
        fig.tight_layout(rect=(0,0,1,.88));save(fig,'yaw_square_circle')
    (output/'provenance.json').write_text(json.dumps({'source_sha256':hashes,'seed':42,'note':'Current target simulator; star FT has paired data in Moderate only. No new simulations.'},indent=2))
    print(output)

if __name__=='__main__':generate()
