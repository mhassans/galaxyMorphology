import yaml
from pathlib import Path
import argparse


def load_config(file_path = None):
    path = Path(file_path)
    with open(path, 'r') as file:
        cfg = yaml.load(file, Loader = yaml.FullLoader)
    #returns the model and the name of the config run
    return cfg, path.stem

def parse_args():
    parser = argparse.ArgumentParser(description = 'Load config file for SVM classification.')
    parser.add_argument('config', type = str, help = 'Path to config file.')

    return parser.parse_args()

