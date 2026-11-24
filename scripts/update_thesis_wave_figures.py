"""Regenerate Chapter 2 wave illustrations from the current recorded W1 contract."""
from pathlib import Path
import importlib.util,sys,json,shutil,hashlib
import yaml,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from bluerov2_gym.envs.core.dynamics import Dynamics
from bluerov2_gym.workflows.plotting.thesis_wave_field import generate

def main():
    source=ROOT/'artifacts/evaluation/article_scale4_four_controllers/benchmark'
    output=ROOT/'artifacts/figures/thesis_current_wave_model'
    spec=importlib.util.spec_from_file_location('article_wave_figures',ROOT/'documents/oceans2026/wave_figures.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    # Route the renderer's publication exports to the diagnostic output, never the article.
    module.PAPER=output/'publication_exports'
    provenance_source=module.PAPER/'reference_sources/plots_master_figures.m'
    provenance_source.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(ROOT/'documents/oceans2026/reference_sources/plots_master_figures.m',provenance_source)
    emitted=module.generate(source,'walker_w1',42,output)
    target=ROOT/'documents/msc_thesis/images/path_tracking/chapter2_wave';target.mkdir(exist_ok=True,parents=True)
    for image in emitted:shutil.copy2(image,target/image.name)
    provenance_path=output/'provenance.json';provenance=json.loads(provenance_path.read_text())
    provenance['source_sha256']['scripts/update_thesis_wave_figures.py']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    for image in emitted:provenance['output_sha256'][image.name]=hashlib.sha256(image.read_bytes()).hexdigest()
    provenance_path.write_text(json.dumps(provenance,indent=2)+'\n')
    shutil.copy2(provenance_path,target/'provenance.json')
    generate(source/'config.resolved.yaml','walker_w1')
    cfg=yaml.safe_load((source/'config.resolved.yaml').read_text());params=cfg['disturbance']['scenarios']['walker_w1']
    model=Dynamics(jonswap_params=params,dynamics_config=cfg['environment']['dynamics_config'])
    record=dict(mechanics_model=model.mechanics_model,total_mass_kg=model.m,displaced_volume_m3=model.volume,weight_N=model.W,buoyancy_N=model.B_force,gravity_center_m=model.r_g.tolist(),buoyancy_center_m=model.r_b.tolist(),inertia_origin_kgm2=model.I_origin.tolist(),disturbance_model=model.disturbance_model,wave_cases={key:{name:cfg['disturbance']['scenarios'][key][name] for name in ['Hs','Tp','water_depth','z_eval','noise_std','scale','max_current']} for key in ['walker_w1','walker_w2','walker_w3']})
    (output/'dynamics_parameters_review.json').write_text(json.dumps(record,indent=2))
    print(json.dumps(record,indent=2))
if __name__=='__main__':main()
