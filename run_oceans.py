"""Compatibility shortcut for the original five-seed curriculum command."""
import sys
from pathlib import Path
from bluerov2_gym.workflows.cli import main
if __name__ == '__main__':
    main(['train','--config',str(Path(__file__).resolve().parent/'configs/training/curriculum.yaml'),
          '--seeds','42','43','44','45','46',*sys.argv[1:]])
