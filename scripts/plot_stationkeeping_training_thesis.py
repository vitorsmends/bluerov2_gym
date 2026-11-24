"""Render retained stationkeeping training history in English and export to thesis."""
from pathlib import Path
import sys,shutil
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import matplotlib
matplotlib.use('Agg')
import yaml
from bluerov2_gym.workflows.plotting.training.thesis import CurriculumFigures


def main():
    history=ROOT/'artifacts/analysis/stationkeeping_training_history'
    if not (history/'lineage.json').is_file():
        raise FileNotFoundError('Build the retained history with training.lineage first.')
    output=history/'figures/training_thesis_en';tables=history/'tables/training_thesis_en'
    settings=dict(log_dir=str(history/'logs/tensorboard'),output_dir=str(output),table_dir=str(tables),
        language='en',font_size=14,legend_font_size=10,smooth_window=10,
        training_config=yaml.safe_load((history/'config.resolved.yaml').read_text()),
        curriculum_status=yaml.safe_load((history/'models/curriculum_status.yaml').read_text()))
    CurriculumFigures(settings).generate()
    dest=ROOT/'documents/msc_thesis/images/stationkeeping/training_current'
    dest.mkdir(parents=True,exist_ok=True)
    for path in output.iterdir():
        if path.suffix in {'.pdf','.png','.md'}:shutil.copy2(path,dest/path.name)
    target=ROOT/'documents/msc_thesis/tables/stationkeeping/training_current'
    target.mkdir(parents=True,exist_ok=True)
    for path in tables.iterdir():shutil.copy2(path,target/path.name)
    shutil.copy2(history/'lineage.json',target/'lineage.json')
    print('English training figures exported to',dest)

if __name__=='__main__':main()
