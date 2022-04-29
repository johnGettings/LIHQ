def ESRGAN_loop(vidFiles):
  #os.cd('/content/BasicSR')
  for vidPath in vidFiles:
    vidName = ntpath.basename(vidPath)[:-4]
    os.mkdir(f'/content/BasicSR/Out/Round1/{vidName}')
    framesOutV2F = '/content/vid2Frames/Round1/' + vidName
    framesOutESR = '/content/BasicSR/Out/Round1/' + vidName
    os.system('python inference/inference_esrgan.py --model_path experiments/pretrained_models/ESRGAN/ESRGAN_SRx4_DF2KOST_official-ff704c30.pth --input {framesOutV2F} --output {framesOutESR}')

def ESRGAN_round2():
  #os.cd('/content/BasicSR')
  for vidPath in vidFiles:
    vidName = ntpath.basename(vidPath)[:-4]
    os.mkdir(f'/content/BasicSR/Out/Round2/{vidName}')
    framesOutV2F = '/content/vid2Frames/Round2/' + vidName
    framesOutESR = '/content/BasicSR/Out/Round2/' + vidName
    os.system('python inference/inference_esrgan.py --model_path experiments/pretrained_models/ESRGAN/ESRGAN_SRx4_DF2KOST_official-ff704c30.pth --input {framesOutV2F} --output {framesOutESR}')