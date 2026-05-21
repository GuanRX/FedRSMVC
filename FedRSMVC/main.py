# main.py
import json
import os
import torch
import random
import numpy as np
import warnings
from Flsimulator import FLSimulator

warnings.filterwarnings('ignore')

def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def load_config(path='config.json'):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found at {path}")
    with open(path, 'r') as f:
        return json.load(f)

if __name__ == "__main__":
    config = load_config()
    seed = config.get('seed', 42)
    set_seed(seed)
    gpu_id = config.get('gpu_id', '0')
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")

    simulator = FLSimulator(config, device=device)
    simulator.start()