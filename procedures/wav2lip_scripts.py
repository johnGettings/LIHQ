import os
import glob
import configparser
import subprocess
import sys

#####Fix all paths below
def wav2lip_run(dir):
  vidPath = f'./Output/FOMM/Round1/{dir}.mp4'
  audPath = f'./input/audio/{dir}/{dir}.wav'
  outPath = f'/content/Wav2Lip/output/{dir}.mp4'
  os.chdir('Wav2Lip')
  command = f'python inference.py --checkpoint_path ./checkpoints/wav2lip_gan.pth --face {vidPath} --audio {audPath} --outfile {outPath}  --pads 0 30 0 0'
  try:
    subprocess.call(command, shell=True)
  except subprocess.CalledProcessError:
    print('!!!!!!! Error with Wav2Lip Paths !!!!!!')
    sys.exit()
  os.chdir('..')

'''   
def wav2lip_config():
    config2 = configparser.ConfigParser()
    
    config2.checkpoint_path = './Wav2Lip/checkpoints/wav2lip_gan.pth'
    config2.face = vidPath
    config2.audio = glob.glob(f'/content/audio/{dir}/*')[0]
    config2.outfile = './Wav2Lip/output/{dir}.mp4'
    config2.pads = [0, 30, 0, 0]
    config2.img_size = 96
    return(config2)
'''