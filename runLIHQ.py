import os
import glob
import ntpath
import shutil
from pathlib import Path
import time

from procedures.av_scripts import *
os.chdir('./first_order_model')
from LIHQ.procedures.fomm_scripts import FOMM_chop_refvid, FOMM_run
os.chdir('..')
from procedures.wav2lip_scripts import wav2lip_run, wav2lip_config
from procedures.esrgan_scripts import ESRGAN_run
os.chdir('./GPEN')
from procedures.gpen_scripts import GPEN_run
os.chdir('..')
from first_order_model.demo import load_checkpoints
from procedures.qvi_scripts import qvi_config
os.chdir('./QVI')
from QVI.demo import main as qvi_main
os.chdir('..')



def run(face, save_path, ref_vid, ref_vid_offset = [0], second_run = True, frame_int = 2, clear_outputs=False):
  
  ##### Error catching
  #if save path is a real path
  #make sure face and refvidoffset are arrays
  
  # auddirnames is list of names of each folder in 'audio' folder = [Folder1, Folder2, Folder3]
  # Each audio folder should have one audio file, for one output video each dir in auddirnames
  
  
  if clear_outputs == True:
    for path in Path("./LIHQ/output").glob("**/*"):
        if path.is_file():
            path.unlink()

  R1start = time.time()
  
  #A/V Set up
  print("Performing Set Up")
  auddirnames = get_auddirnames()
  for dir in auddirnames:
    combine_audiofiles(dir)
  
  #expanding face array as needed
  while len(face) < len(auddirnames):
    face.append(face[0])
  
  #FOMM
  #Cropping reference video
  FOMM_chop_refvid(auddirnames, ref_vid, ref_vid_offset)
  
  #Running FOMM
  print("Running First Order Motion Model")
  generator, kp_detector = load_checkpoints(config_path='./first_order_model/config/vox-256.yaml', checkpoint_path='./first_order_model/vox-cpk.pth.tar')
  i = 0
  for dir in auddirnames:
    sub_clip = f'./first_order_model/input-ref-vid/{dir}/{dir}.mp4'
    FOMM_run(face[i], sub_clip, generator, kp_detector, dir, Round = "1")
    i+=1
  print("FOMM Success!")

  #Wav2Lip
  print("Running Wav2Lip")
  for dir in auddirnames:
    #config2 = wav2lip_config()
    wav2lip_run(dir)
  print("Wav2Lip Success!")

  #Vid 2 Frames
  for dir in auddirnames:
    framesOutV2F = f'./output/vid2Frames/Round1/{dir}/'
    vidPath = f'./Wav2Lip/output/{dir}.mp4'
    os.mkdir(framesOutV2F)
    vid2frames(vidPath, framesOutV2F)

  #ESRGAN
  print("Performing Upscaling and Restoration!")
  for dir in auddirnames:
    ESRGAN_run(dir, Round = "1")

  #GPEN
  #Play around with SR=True/False
  for dir in auddirnames:
    GPEN_run(dir, Round = "1")

  #franme2Vid
  for dir in auddirnames:
    frames2vid(dir, Round = "1")
  
  #Round1 Printouts
  print("Round 1 Complete!")
  R1end = time.time()
  print("Round1 Elapsed Time:")
  print(R1end - R1start)

  if second_run == True:
    print("Beginning Round 2")
    R2start = time.time()
    #FOMM Round 2
    i=0
    for dir in auddirnames:
      refVideo = f'./GPEN/Out/VidOutR1/{dir}.mp4'
      FOMM_run(face[i], refVideo, dir, Round = "2", relativeTF = False)
      i+=1

    #Vid2Frames R2
    for dir in auddirnames:
      framesOutV2F = f'./output/vid2Frames/Round2/{dir}/'
      vidPath = f'./Output/FOMM/Round2/{dir}.mp4'
      os.mkdir(framesOutV2F)
      vid2frames(vidPath, framesOutV2F)

    #ESRGAN R2
    for dir in auddirnames:
      ESRGAN_run(dir, Round = "2")

    #GPEN R2
    for dir in auddirnames:
      GPEN_run(dir, Round = "2")

    #Final Frames2Vid
    for dir in auddirnames:
      frames2vid(dir, Round = "2")
    
    R2end = time.time()
    print("Round2 Elapsed Time")
    print(R2end - R2start)
  

  # QVI frame interpolation
  if frame_int != None:
    print('Beginning Frame Interpolation.')
    QVIstart = time.time()
    for dir in auddirnames:
      config = qvi_config(dir, frame_int)
      qvi_main(config)
    
    print('Frame Interpolation Complete!')
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
  