import glob
import wave
import os
import subprocess
import sys

import cv2


#Get a list of all subfolders in audio directory
def get_auddirnames(audio_super):
  audfolders = sorted(glob.glob(audio_super + "*/"))
  if len(audfolders) < 1:
    print('Check your audio folder: ' + audio_super)
    sys.exit()
  auddirnames = []
  for ff in audfolders:
    auddirnames.append(os.path.basename(os.path.dirname(ff)))
  return auddirnames

#audiofiles should be in numerical order
#removes everything except combined audio. Coimbined audio name is same as subfolder.
def combine_audiofiles(dir, audio_super):
  audioFiles = sorted(glob.glob(f'{audio_super}{dir}/*'))
  
  if len(audioFiles) > 1:
    outfile = f'{audio_super}{dir}/{dir}.wav'
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
      os.remove(filez)

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
def frames2vid(fps, audPath, framesInPath, vidOutPath):
    command = f'ffmpeg -y -r {fps} -f image2 -i {framesInPath} -i {audPath} -vcodec mpeg4 -b:v 20000k {vidOutPath}'
    try:
      subprocess.call(command, shell=True)
    except subprocess.CalledProcessError:
      print('!!!!!!! Error converting frames back to video. !!!!!!')
      sys.exit()