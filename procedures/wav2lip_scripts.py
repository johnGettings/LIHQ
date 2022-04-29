import os
import glob

def Wav2Lip_loop():
  for dir in auddirnames:
    vidFiles = glob.glob('/content/first-order-model/output/{}/*'.format(dir))
    for vidPath in vidFiles:
      audPath = glob.glob('/content/audio/{}/*'.format(dir))[0]
      outPath = '/content/Wav2Lip/output/' + dir + '.mp4'
      os.system('python inference.py --checkpoint_path checkpoints/wav2lip_gan.pth --face {vidPath} --audio {audPath} --outfile {outPath}  --pads 0 30 0 0')