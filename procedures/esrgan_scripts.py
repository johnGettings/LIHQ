import os
import ntpath

def ESRGAN_run(vidPath, Round):
    vidName = ntpath.basename(vidPath)[:-4]
    os.mkdir(f'./BasicSR/Out/Round{Round}/{vidName}')
    framesOutV2F = f'./vid2Frames/Round{Round}/' + vidName
    framesOutESR = f'./BasicSR/Out/Round{Round}/' + vidName
    os.system('python inference/inference_esrgan.py --model_path experiments/pretrained_models/ESRGAN/ESRGAN_SRx4_DF2KOST_official-ff704c30.pth --input {framesOutV2F} --output {framesOutESR}')