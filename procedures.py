################################################  VITS
import matplotlib.pyplot as plt
import IPython.display as ipd

import sys
import os
import cv2
import json
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from pydub import AudioSegment

import commons
import utils
from data_utils import TextAudioLoader, TextAudioCollate, TextAudioSpeakerLoader, TextAudioSpeakerCollate
from models import SynthesizerTrn
from text.symbols import symbols
from text import text_to_sequence

from scipy.io.wavfile import write

#Functions
def get_text(text, hps):
    %cd /content/vits
    text_norm = text_to_sequence(text, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm

def VITS_load_checkpoint():
  %cd /content/vits
  hps = utils.get_hparams_from_file("/content/vits/configs/vctk_base.json")

  net_g = SynthesizerTrn(
     len(symbols),
     hps.data.filter_length // 2 + 1,
     hps.train.segment_size // hps.data.hop_length,
     n_speakers=hps.data.n_speakers,
     **hps.model).cuda()
  _ = net_g.eval()

  _ = utils.load_checkpoint("/content/drive/MyDrive/ML/pretrained_vctk.pth", net_g, None)

  return hps, net_g

def VITS_run_inference(hps, net_g, speaker, speechText):
  %cd /content/vits
  stn_tst = get_text(speechText, hps)
  with torch.no_grad():
    x_tst = stn_tst.cuda().unsqueeze(0)
    x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).cuda()
    sid = torch.LongTensor([speaker]).cuda()
    audio = net_g.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=.85, noise_scale_w=0.8, length_scale=1.1)[0][0,0].data.cpu().float().numpy()
  ipd.display(ipd.Audio(audio, rate=hps.data.sampling_rate))

  return audio

def VITS_save_audio(audio, folderName, fileName):
  audio = ipd.Audio(audio, rate=hps.data.sampling_rate)
  audio = AudioSegment(audio.data, frame_rate=22050, sample_width=2, channels=1)
  folderPath = '/content/audio/' + folderName 
  %mkdir '{folderPath}'
  savePath = '/content/audio/' + folderName + '/' + fileName
  audio.export(savePath, format="mp3", bitrate="64k")

################################################  A/V Setup

import glob
import wave

#Get a list of all subfolders in audio directory
def get_auddirnames():
  audfolders = sorted(glob.glob("/content/audio/*/"))
  global auddirnames
  auddirnames = []
  for ff in audfolders:
    auddirnames.append(os.path.basename(os.path.dirname(ff)))
  return auddirnames

#audiofiles should be in numerical order
#removes everything except combined audio. Coimbined audio name is same as subfolder.
def combine_audiofiles():
  for dir in auddirnames:
    audioFiles = sorted(glob.glob('/content/audio/{}/*'.format(dir)))
    
    if len(audioFiles) > 1:
      outfile = '/content/audio/{}/{}.wav'.format(dir,dir)
      data= []
      for infile in audioFiles:
          w = wave.open(infile, 'rb')
          data.append( [w.getparams(), w.readframes(w.getnframes())] )
          w.close()
      output = wave.open(outfile, 'wb')
      output.setparams(data[0][0])
      for i in range(len(data)):
          output.writeframes(data[i][1])
      output.close()

      for filez in audioFiles:
        %rm '{filez}'

# converts video to frames; outpouts in 0000x.png to framesOutPath location
def vid2frames(vidPath, framesOutPath):
    vidcap = cv2.VideoCapture(vidPath)
    success,image = vidcap.read()
    frame = 1
    while success:
      cv2.imwrite(framesOutPath + str(frame).zfill(5) + '.png', image)
      success,image = vidcap.read()
      frame += 1

#Merging back into video
def Frame2VidR1():
  for dir in auddirnames:
    audPath = glob.glob('/content/audio/{}/*'.format(dir))[0]
    frames = '/content/GPEN/Out/Round1/' + dir + '/*.png'
    outPath = '/content/GPEN/Out/VidOutR1/' + dir + '.mp4'
    !ffmpeg -y -r 25 -f image2 -pattern_type glob -i '{frames}' -i '{audPath}' -vcodec mpeg4 -b 1500k '{outPath}'

def Frame2VidR2():
  for dir in auddirnames:
    audPath = glob.glob('/content/audio/{}/*'.format(dir))[0]
    frames = '/content/GPEN/Out/Round2/' + dir + '/*.png'
    outPath = '/content/FinalVideos/' + faceName + '.mp4'
    !ffmpeg -y -r 25 -f image2 -pattern_type glob -i '{frames}' -i '{audPath}' -vcodec mpeg4 -b 1500k '{outPath}'

################################################  FOMM
import imageio
import numpy as np
import librosa
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from skimage.transform import resize
from IPython.display import HTML
from demo import make_animation
from skimage import img_as_ubyte
import ntpath
import warnings
import moviepy.editor as mpe
from moviepy.video.io.VideoFileClip import VideoFileClip


def FOMM_chop_refvid(refVid, refVidOffset = 0):
  # Create chopped ref vids into segments the same length as the audio
  totalAudioLength = 0 + refVidOffset
  for dir in auddirnames:
    %mkdir '/content/first-order-model/input-ref-vid/{dir}'
    audio = glob.glob('/content/audio/{}/*'.format(dir))[0]
    audioLength = librosa.get_duration(filename = audio)

    input_video_path = refVid
    output_video_path = '/content/first-order-model/input-ref-vid/{}/{}.mp4'.format(dir, dir)
    with VideoFileClip(input_video_path) as video:
      new = video.subclip(totalAudioLength, totalAudioLength + audioLength)
      new.write_videofile(output_video_path, audio_codec='aac')

    totalAudioLength = totalAudioLength + audioLength

#Output is in first-order-model folder (output or output-with-sound)
def FOMM_run(SourceImgPath, SourceVidPath, dir, relativeTF):
  %cd '/content/first-order-model'
  warnings.filterwarnings("ignore")

  source_image = imageio.imread(SourceImgPath)
  reader = imageio.get_reader(SourceVidPath)

  #Resize image and video to 256x256
  source_image = resize(source_image, (256, 256))[..., :3]

  fps = reader.get_meta_data()['fps']
  driving_video = []
  try:
      for im in reader:
          driving_video.append(im)
  except RuntimeError:
      pass
  reader.close()

  driving_video = [resize(frame, (256, 256))[..., :3] for frame in driving_video]

  predictions = make_animation(source_image, driving_video, generator, kp_detector, relative = relativeTF)
  vidPath = SourceVidPath

  #save resulting video
  FOMMoutPath = '/content/first-order-model/output/{}/'.format(dir) + ntpath.basename(vidPath)[:-4] + '.mp4'
  imageio.mimsave(FOMMoutPath, [img_as_ubyte(frame) for frame in predictions], fps=fps)

def FOMM_loop(refImg, relativeTF):
  #Running FOMM
  for dir in auddirnames:
    %mkdir '/content/first-order-model/output/{dir}'
    vidFiles = glob.glob('/content/first-order-model/input-ref-vid/{}/{}.mp4'.format(dir, dir))
    for vidPath in vidFiles:
      FOMM_run(refImg, vidPath, dir, relativeTF)

################################################  Wav2Lip
def Wav2Lip_loop():
  %cd /content/Wav2Lip
  for dir in auddirnames:
    vidFiles = glob.glob('/content/first-order-model/output/{}/*'.format(dir))
    for vidPath in vidFiles:
      audPath = glob.glob('/content/audio/{}/*'.format(dir))[0]
      outPath = '/content/Wav2Lip/output/' + dir + '.mp4'
      !python inference.py --checkpoint_path checkpoints/wav2lip_gan.pth --face '{vidPath}' --audio '{audPath}' --outfile '{outPath}'  --pads 0 30 0 0

################################################  ESRGAN
def ESRGAN_loop(vidFiles):
  %cd /content/BasicSR
  for vidPath in vidFiles:
    vidName = ntpath.basename(vidPath)[:-4]
    %mkdir '/content/BasicSR/Out/Round1/{vidName}'
    framesOutV2F = '/content/vid2Frames/Round1/' + vidName
    framesOutESR = '/content/BasicSR/Out/Round1/' + vidName
    !python inference/inference_esrgan.py --model_path experiments/pretrained_models/ESRGAN/ESRGAN_SRx4_DF2KOST_official-ff704c30.pth --input '{framesOutV2F}' --output '{framesOutESR}'

def ESRGAN_round2():
  %cd /content/BasicSR
  for vidPath in vidFiles:
    vidName = ntpath.basename(vidPath)[:-4]
    %mkdir '/content/BasicSR/Out/Round2/{vidName}'
    framesOutV2F = '/content/vid2Frames/Round2/' + vidName
    framesOutESR = '/content/BasicSR/Out/Round2/' + vidName
    !python inference/inference_esrgan.py --model_path experiments/pretrained_models/ESRGAN/ESRGAN_SRx4_DF2KOST_official-ff704c30.pth --input '{framesOutV2F}' --output '{framesOutESR}'

################################################  GPEN
from face_enhancement import FaceEnhancement

#Functions
def GPEN_loop(Round):
  %cd /content/GPEN
  for vidPath in vidFiles:
    vidName = ntpath.basename(vidPath)[:-4]
    framesOutESR = '/content/BasicSR/Out/{}/'.format(Round) + vidName
    filesIn = glob.glob(framesOutESR + '/*')

    for filex in filesIn:
      #Setting directories
      base_num = ntpath.basename(filex)[:-11]
      
      if __name__=="__main__":    
          model = {'name':'GPEN-BFR-512', 'size':512, 'channel_multiplier':2, 'narrow':1, 'use_cuda':True}
          
          outdir = '/content/GPEN/Out/{}/'.format(Round) + vidName + '/'
          os.makedirs(outdir, exist_ok=True)

          faceenhancer = FaceEnhancement(use_sr=False, device='cuda', size=model['size'], model=model['name'], channel_multiplier=model['channel_multiplier'], narrow=model['narrow'])
          
          im = cv2.imread(filex, cv2.IMREAD_COLOR) # BGR

          img, orig_faces, enhanced_faces = faceenhancer.process(im)
          
          im = cv2.resize(im, img.shape[:2][::-1])
          cv2.imwrite(outdir + base_num +'.png', img)
          print(base_num)