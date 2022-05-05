import os
import glob
import ntpath
import shutil
from pathlib import Path
import time

from av_scripts import *
from fomm_scrits import FOMM_chop_refvid, FOMM_run
from wav2lip_scripts import Wav2Lip_run
from esrgan_scripts import ESRGAN_run
from gpen_scripts import GPEN_run
from first_order_model.demo import load_checkpoints
from qvi_scripts import qvi_config
from QVI.demo import main as qvi_main




def run(face, save_path, ref_vid, ref_vid_offset = [0], second_run = True, frame_int = 2, clear_outputs=False):
  
  ##### Error catching
  #if save path is a real path
  #make sure face and refvidoffset are arrays
  
  
  if clear_outputs == True:
    for path in Path("./LIHQ/output").glob("**/*"):
        if path.is_file():
            path.unlink()

  R1start = time.time()
  
  #A/V Set up
  auddirnames = get_auddirnames()
  combine_audiofiles(auddirnames)
  
  #FOMM
  #expanding face array if needed
  while len(face) < len(auddirnames):
    face.append(face[0])
  
  #Cropping reference video
  FOMM_chop_refvid(auddirnames, ref_vid, ref_vid_offset)
  
  #Running FOMM
  generator, kp_detector = load_checkpoints(config_path='./config/vox-256.yaml', checkpoint_path='vox-cpk.pth.tar')
  i = 0
  for dir in auddirnames:
    os.mkdir(f'./first_order_model/output/{dir}')
    sub_clip = f'./first_order_model/input-ref-vid/{dir}/{dir}.mp4'
    FOMM_run(face[i], sub_clip, generator, kp_detector, dir)
    i+=1

  #Wav2Lip
  for dir in auddirnames:
    Wav2Lip_run(dir)

  #Vid 2 Frames
  vidFiles = glob.glob('./Wav2Lip/output/*')
  for vidPath in vidFiles:
    framesOutV2F = './output/vid2Frames/Round1/' + ntpath.basename(vidPath)[:-4] + '/'
    os.mkdir(framesOutV2F)
    vid2frames(vidPath, framesOutV2F)

  #ESRGAN
  for vidPath in vidFiles:
    ESRGAN_run(vidPath, Round = "1")

  #GPEN
  #Play around with SR=True/False
  for vidPath in vidFiles:
    GPEN_run(vidPath, Round = "1")

  #franme2Vid
  Frame2Vid(auddirnames, Round = "1")
  
  R1end = time.time()
  print("Round1 Elapsed Time")
  print(R1end - R1start)

  if second_run == True:
    R2start = time.time()
    #FOMM Round 2
    i=0
    for dir in auddirnames:
      refVideo = f'./GPEN/Out/VidOutR1/{dir}.mp4'
      FOMM_run(face[i], refVideo, dir, relativeTF = False)
      i+=1

    #Vid2Frames R2
    for dir in auddirnames:
      vidFiles = glob.glob(f'./first_order_model/output/{dir}/*')
      for vidPath in vidFiles:
        framesOutV2F = './output/vid2Frames/Round2/' + ntpath.basename(vidPath)[:-4] + '/'
        os.mkdir(framesOutV2F)
        vid2frames(vidPath, framesOutV2F)

    #ESRGAN R2
    for vidPath in vidFiles:
      ESRGAN_run(vidPath, Round = "2")

    #GPEN R2
    for vidPath in vidFiles:
      GPEN_run(vidPath, Round = "2")

    #Final Frames2Vid
    Frame2Vid(auddirnames, Round = "2")
    
    R2end = time.time()
    print("Round2 Elapsed Time")
    print(R2end - R2start)
  

  # QVI frame interpolation
  if frame_int != None:
    QVIstart = time.time()
    for dir in auddirnames:
      config = qvi_config(dir, frame_int)
      qvi_main(config)
    
    QVIend = time.time()
    print("QVI Elapsed Time")
    print(QVIend - QVIstart)
  
  #Copying final video to out location
  if frame_int != None:
    for dir in auddirnames:
      src = f'./QVI/{dir}.mp4'
      shutil.copyfile(src, save_path)
  else: 
    if second_run == False:
      Round = 1
    else:
      Round = 2
    for dir in auddirnames:
      src = f'./GPEN/Out/VidOutR{Round}/{dir}.mp4'
      shutil.copyfile(src, save_path)
  