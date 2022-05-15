import glob
import wave
import os
import subprocess
import sys

import cv2


#Get a list of all subfolders in audio directory
def get_auddirnames():
  audfolders = sorted(glob.glob("./input/audio/*/"))
  auddirnames = []
  for ff in audfolders:
    auddirnames.append(os.path.basename(os.path.dirname(ff)))
  return auddirnames

#audiofiles should be in numerical order
#removes everything except combined audio. Coimbined audio name is same as subfolder.
def combine_audiofiles(dir):
  audioFiles = sorted(glob.glob(f'./input/audio/{dir}/*'))
  
  if len(audioFiles) > 1:
    outfile = f'./input/audio/{dir}/{dir}.wav'
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
def frames2vid(dir, Round):
    audPath = glob.glob(f'./input/audio/{dir}/*')[0]
    frames = f'./GPEN/Out/Round{Round}/{dir}/*.png'
    outPath = f'./GPEN/Out/VidOutR{Round}/{dir}.mp4'
    command = f'ffmpeg -y -r 25 -f image2 -pattern_type glob -i {frames} -i {audPath} -vcodec mpeg4 -b 20000k {outPath}'
    try:
      subprocess.call(command, shell=True)
    except subprocess.CalledProcessError:
      print('!!!!!!! Error converting frames back to video. !!!!!!')
      sys.exit()