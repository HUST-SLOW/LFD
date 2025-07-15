import os
import sys
sys.path.append(os.getcwd())
import argparse
import warnings
warnings.filterwarnings('ignore')
from utils.load_config import load_yaml
from utils.utils import setup_seed
from models._LFD import LFD

def get_args():
    parser = argparse.ArgumentParser(description='LFD mvtec/mvtec 3d/visa supervised segmentation')
    parser.add_argument('--config', type=str, default='configs/LFD.yaml', help='config file path')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    setup_seed(42)
    args = get_args()
    cfg = load_yaml(args.config)
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg['device']
    print(cfg)

    os.makedirs(cfg['outputs']['ckpt_path'], exist_ok=True)
    os.makedirs(cfg['outputs']['vis_path'], exist_ok=True)
    
    model = LFD(cfg)
    if cfg['training']['state']:
        model.train()
    if cfg['testing']['state']:
        model.test()
    if cfg['eval_speed']['state']:   
        model.eval_fps()

