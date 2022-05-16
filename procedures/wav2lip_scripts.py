import os
import subprocess
import sys

from pathlib import Path

def wav2lip_run(dir):
  vidPath = f'{os.getcwd()}/output/FOMM/Round1/{dir}.mp4'
  audPath = f'{os.getcwd()}/input/audio/{dir}/{dir}.wav'
  outPath = f'{os.getcwd()}/output/wav2Lip/{dir}.mp4'
  os.chdir('Wav2Lip')
  command = f'python inference.py --checkpoint_path ./checkpoints/wav2lip_gan.pth --face {vidPath} --audio {audPath} --outfile {outPath}  --pads 0 30 0 0'
  try:
    subprocess.call(command, shell=True)
  except subprocess.CalledProcessError:
    print('!!!!!!! Error with Wav2Lip Paths !!!!!!')
    sys.exit()
  os.chdir('..')