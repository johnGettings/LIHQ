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
from procedures.wav2lip_scripts import wav2lip_run
from first_order_model.demo import load_checkpoints
from procedures.qvi_scripts import qvi_config
os.chdir('./QVI')
from LIHQ.QVI.demo import main as qvi_main
os.chdir('..')



def run(face, save_path = None, audio_super = '/content/LIHQ/input/audio/', ref_vid = '/content/LIHQ/input/ref_vid/syn_reference.mp4', ref_vid_offset = [0], frame_int = 3, clear_outputs=False):
  
  ##### Error catching
  #if save path is a real path
  #make sure face and refvidoffset are arrays
  
  # auddirnames is list of names of each folder in 'audio' folder = [Folder1, Folder2, Folder3]
  # Each audio folder should have one audio file, for one output video each dir in auddirnames
  if frame_int is not None:
    fps = 25 * (frame_int + 1)
  else:
    fps = 25
  
  if clear_outputs == True:
    for path in Path("./output").glob("**/*"):
        if path.is_file():
            path.unlink()

  R1start = time.time()
  
  #A/V Set up
  print("Performing Set Up")
  if audio_super[-1:] != '/':
    audio_super = audio_super + '/'
  auddirnames = get_auddirnames(audio_super)
  for dir in auddirnames:
    combine_audiofiles(dir, audio_super)
  
    #expanding face array as needed
  while len(face) < len(auddirnames):
    face.append(face[0])
  
  #FOMM
    #Cropping reference video
  FOMM_chop_refvid(auddirnames, ref_vid, audio_super, ref_vid_offset)
  
    #Running FOMM (Mimicking facial movements from reference video)
  print("Running First Order Motion Model")
  generator, kp_detector = load_checkpoints(config_path='./first_order_model/config/vox-256.yaml', checkpoint_path='./first_order_model/vox-cpk.pth.tar')
  i = 0
  for dir in auddirnames:
    sub_clip = f'./first_order_model/input-ref-vid/{dir}/{dir}.mp4'
    FOMM_run(face[i], sub_clip, generator, kp_detector, dir, Round = "1")
    i+=1
  print("FOMM Success!")

  #Wav2Lip (Generating lip movement from audio)
  print("Running Wav2Lip")
  for dir in auddirnames:
    wav2lip_run(dir)
  print("Wav2Lip Success!")

  #Vid 2 Frames (Converting wav2Lip output to frames for next step)
  for dir in auddirnames:
    framesOutV2F = f'./output/vid2Frames/Round1/{dir}/'
    vidPath = f'./output/wav2Lip/{dir}.mp4'
    os.makedirs(framesOutV2F, exist_ok=True)
    vid2frames(vidPath, framesOutV2F)

  #GFPGAN (Restoration and upscaling)
  print("Beginning restoration and upscaling")
  os.chdir('GFPGAN')
  in_pth = str(Path(os.getcwd()).parent.absolute()) + f'/output/vid2Frames/Round1/{dir}/'
  out_pth = str(Path(os.getcwd()).parent.absolute()) + f'/output/GFPGAN/Round1/{dir}/'
  for dir in auddirnames:
    command = f'python inference_gfpgan.py -i {in_pth} -o {out_pth} -v 1.3 -s 4 --bg_upsampler realesrgan'
    try:
      subprocess.call(command, shell=True)
    except subprocess.CalledProcessError:
      print('!!!!!!! Error with GFPGAN command !!!!!!')
      sys.exit()
  os.chdir('..')
  print('Completed Restoration Round 1')

  #frames2Vid (Converting frames back to video)
  for dir in auddirnames:
    audPath = glob.glob(f'{audio_super}{dir}/*')[0]
    framesInPath = f'./output/GFPGAN/Round1/{dir}/restored_imgs/%5d.png'
    vidOutPath = f'./output/frames2Vid/Round1/{dir}.mp4'
    frames2vid(25, audPath, framesInPath, vidOutPath)
  
  #Round1 Printouts
  print("Round 1 Complete!")
  R1end = time.time()
  print("Round1 Elapsed Time:")
  print(R1end - R1start)

 #### Round 2
  print("Beginning Round 2")
  R2start = time.time()
  
  #FOMM Round 2
  i=0
  for dir in auddirnames:
    refVideo = f'./output/frames2Vid/Round1/{dir}.mp4'
    FOMM_run(face[i], refVideo, generator, kp_detector, dir, Round = "2", relativeTF = False)
    i+=1

  #Vid2Frames R2
  for dir in auddirnames:
    framesOutV2F = f'./output/vid2Frames/Round2/{dir}/'
    vidPath = f'./output/FOMM/Round2/{dir}.mp4'
    os.makedirs(framesOutV2F, exist_ok=True)
    vid2frames(vidPath, framesOutV2F)

  #GFPGAN (Restoration and upscaling)
  os.chdir('GFPGAN')
  in_pth = str(Path(os.getcwd()).parent.absolute()) + f'/output/vid2Frames/Round2/{dir}/'
  out_pth = str(Path(os.getcwd()).parent.absolute()) + f'/output/GFPGAN/Round2/{dir}/'
  for dir in auddirnames:
    command = f'python inference_gfpgan.py -i {in_pth} -o {out_pth} -v 1.3 -s 4 --bg_upsampler realesrgan'
    try:
      subprocess.call(command, shell=True)
    except subprocess.CalledProcessError:
      print('!!!!!!! Error with GFPGAN command !!!!!!')
      sys.exit()
  os.chdir('..')
  
  R2end = time.time()
  print("Round2 Elapsed Time")
  print(R2end - R2start)
  

  if frame_int == None:
    #Final Frames2Vid
    for dir in auddirnames:
      audPath = glob.glob(f'{audio_super}{dir}/*')[0]
      framesInPath = f'./output/GFPGAN/Round2/{dir}/restored_imgs/%5d.png'
      vidOutPath = f'./output/frames2Vid/Round2/{dir}.mp4'
      frames2vid(25, audPath, framesInPath, vidOutPath)
      
  else:  
    # QVI (Frame interpolation)
    print('Beginning Frame Interpolation.')
    QVIstart = time.time()
    for dir in auddirnames:
      os.makedirs(f'./output/QVI/{dir}/', exist_ok=True)
      config = qvi_config(dir, frame_int)
      os.chdir('QVI')
      qvi_main(config)
      os.chdir('..')
    
      audPath = glob.glob(f'{audio_super}{dir}/*')[0]
      framesInPath = f'./output/GFPGAN/Round2/{dir}/restored_imgs/%5d.png'
      vidOutPath = f'./output/frames2Vid/Round2/{dir}.mp4'
      frames2vid(fps, audPath, framesInPath, vidOutPath)
    
    print('Frame Interpolation Complete!')
    QVIend = time.time()
    print("QVI Elapsed Time")
    print(QVIend - QVIstart)
  
  #Copying to final vids folder
  src = f'./output/frames2Vid/Round2/{dir}.mp4'
  final_vids = f'./output/finalVidsOut/{dir}.mp4'
  shutil.copyfile(src, final_vids)
  
  #Copying final video to save_path
  if save_path != None:
    for dir in auddirnames:
      src = f'./output/finalVidsOut/{dir}.mp4'
      shutil.copyfile(src, save_path)
  