import os
import glob

def Wav2Lip_run(dir):
  vidFiles = glob.glob(f'/content/first-order-model/output/{dir}/*')
  for vidPath in vidFiles:
    audPath = glob.glob(f'/content/audio/{dir}/*')[0]
    outPath = '/content/Wav2Lip/output/{dir}.mp4'
    os.system('python inference.py --checkpoint_path checkpoints/wav2lip_gan.pth --face {vidPath} --audio {audPath} --outfile {outPath}  --pads 0 30 0 0')