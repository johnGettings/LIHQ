import glob
import os
import subprocess
import sys

import cv2
from pydub import AudioSegment


#Get a list of all subfolders in audio directory
def get_auddirnames(audio_super):
  aud_folders = sorted(glob.glob(audio_super + "*/"))
  if len(aud_folders) < 1:
    print('No audio folders. Check your parent audio folder: ' + audio_super)
    sys.exit()
  aud_dir_names = []
  for ff in aud_folders:
    aud_dir_names.append(os.path.basename(os.path.dirname(ff)))
  return aud_dir_names

#audiofiles should be in numerical/ alphabetical order
#removes everything except combined audio. Coimbined audio name is same as subfolder.
def combine_audiofiles(adir, audio_super):
  audio_files = sorted(glob.glob(f'{audio_super}{adir}/*'))
  
  #If is more than one audio file, combine them
  if len(audio_files) > 1:
    speech = AudioSegment.from_wav(audio_files[0])
    
    for i in range(len(audio_files)-1):
      if i > 0:
        speech_n = AudioSegment.from_wav(audio_files[i])
        speech = speech + speech_n
    
    speech.export(f'{audio_super}{adir}/{adir}.wav', format="wav")

    for filez in audio_files:
      os.remove(filez)
  
  if len(audio_files) == 1:
    os.rename(audio_files[0], str(os.path.dirname(audio_files[0]))+ '/' + str(os.path.basename(os.path.dirname(audio_files[0]))) + audio_files[0][-4:])
    
  if len(audio_files) == 0:
    print('Missing audio in your audio folder.')
    sys.exit()

# converts video to frames; outpouts in 0000x.png to framesOutPath location
def vid2frames(vid_path, frames_out_path):
    vidcap = cv2.VideoCapture(vid_path)
    success,image = vidcap.read()
    frame = 1
    while success:
      cv2.imwrite(frames_out_path + str(frame).zfill(5) + '.png', image)
      success,image = vidcap.read()
      frame += 1

#Merging back into video
def frames2vid(fps, aud_path, frames_in_path, vid_out_path):
    command = f'ffmpeg -y -r {fps} -f image2 -i {frames_in_path} -i {aud_path} -vcodec mpeg4 -b:v 20000k {vid_out_path}'
    try:
      subprocess.call(command, shell=True)
    except subprocess.CalledProcessError:
      print('!!!!!!! Error converting frames back to video. !!!!!!')
      sys.exit()