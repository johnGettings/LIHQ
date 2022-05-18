import configparser
from pathlib import Path
import os

def qvi_config(dir, frame_int):
    lihq_path = str(os.getcwd())
    config = configparser.ConfigParser()
    config.testset_root = lihq_path + f'/output/GFPGAN/Round2/{dir}/'
    config.test_size = (1024, 1024)
    config.test_crop_size = (1024, 1024)

    config.mean = [0.429, 0.431, 0.397]
    config.std  = [1, 1, 1]

    config.inter_frames = frame_int


    config.model = 'QVI'
    config.pwc_path = './utils/pwc-checkpoint.pt'


    config.store_path = lihq_path + f'/output/QVI/{dir}/'
    config.checkpoint = './model.pt'
    
    return(config)