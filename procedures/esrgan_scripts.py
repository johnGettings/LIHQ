import os
import ntpath
import subprocess
import sys

def ESRGAN_run(dir, Round):
    os.mkdir(f'./BasicSR/Out/Round{Round}/{dir}')
    framesOutV2F = f'./vid2Frames/Round{Round}/{dir}'
    framesOutESR = f'./BasicSR/Out/Round{Round}/{dir}'
    os.chdir('BasicSR')
    command = f'python inference/inference_esrgan.py --model_path experiments/pretrained_models/ESRGAN/ESRGAN_SRx4_DF2KOST_official-ff704c30.pth --input {framesOutV2F} --output {framesOutESR}'
    try:
        subprocess.call(command, shell=True)
    except subprocess.CalledProcessError:
        print('!!!!!!! Error with ESRGAN Paths !!!!!!')
        sys.exit()
    os.chdir('..')