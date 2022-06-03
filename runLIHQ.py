import os
import glob
import shutil
from pathlib import Path
import time
import subprocess
import argparse

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



def run(face, audio_super = '/content/LIHQ/input/audio/', ref_vid = '/content/LIHQ/input/ref_vid/syn_reference.mp4', ref_vid_offset = [0], frame_int = None, clear_outputs=True, save_path = None):

    #Miscellaneous things
    print("Initializing")
      #Turning face &offset to arrays as needed
    if not isinstance(face, list):
        face = [face]
    if not isinstance(ref_vid_offset, list):
        ref_vid_offset = [ref_vid_offset]

      #Determining final fps for ffmpeg
    if frame_int is not None:
        fps = 25 * (frame_int + 1)
    else:
        fps = 25

      #Deleteing output files
    if clear_outputs == True:
        for path in Path("./output").glob("**/*"):
            if path.is_file():
                path.unlink()

      #A/V Set up
    R1start = time.time()
    if audio_super[-1:] != '/':
        audio_super = audio_super + '/'
    aud_dir_names = get_auddirnames(audio_super)
    for adir in aud_dir_names:
        combine_audiofiles(adir, audio_super)

      #Expanding face array as needed
    while len(face) < len(aud_dir_names):
        face.append(face[0])

    #FOMM
      #Cropping reference video
    FOMM_chop_refvid(aud_dir_names, ref_vid, audio_super, ref_vid_offset)

      #Running FOMM (Mimicking facial movements from reference video)
    print("Running First Order Motion Model")
    generator, kp_detector = load_checkpoints(config_path='./first_order_model/config/vox-256.yaml', checkpoint_path='./first_order_model/vox-cpk.pth.tar')
    i = 0
    for adir in aud_dir_names:
        sub_clip = f'./first_order_model/input-ref-vid/{adir}/{adir}.mp4'
        FOMM_run(face[i], sub_clip, generator, kp_detector, adir, Round = "1")
        i+=1
    print("FOMM Success!")

    #Wav2Lip (Generating lip movement from audio)
    print("Running Wav2Lip")
    for adir in aud_dir_names:
        wav2lip_run(adir)
    w2l_folders = sorted(glob.glob('./output/wav2Lip/*'))
    if len(w2l_folders) < len(aud_dir_names):
        print('Wav2Lip could not generate at least one of your videos.\n'
            'Possibly bad audio, unrecognizable mouth, bad file paths, out of memory.\n'
            'Run below command in a separate cell to get full traceback.\n'
            '###########################################################\n'
            '###########################################################\n'
            'import os\n'
            'adir = \'Folder1\' # The audio folder that failed. See Wav2Lip output folder to see whats missing.\n\n'
            'vid_path = f\'''{os.getcwd()}/output/FOMM/Round1/{adir}.mp4\'''\n'
            'aud_path = f\'''{os.getcwd()}/input/audio/{adir}/{adir}.wav\'''\n'
            '%cd /content/LIHQ/Wav2Lip\n'
            '!python inference.py --checkpoint_path ./checkpoints/wav2lip.pth --face {vid_path} --audio {aud_path} --outfile /content/test.mp4  --pads 0 20 0 0\n\n'
            )
        sys.exit()
    else:
        print('Wav2Lip Complete')

    #Vid 2 Frames (Converting wav2Lip output to frames for next step)
    for adir in aud_dir_names:
        frames_out_V2F = f'./output/vid2Frames/Round1/{adir}/'
        vidPath = f'./output/wav2Lip/{adir}.mp4'
        os.makedirs(frames_out_V2F, exist_ok=True)
        vid2frames(vidPath, frames_out_V2F)

    #GFPGAN (Restoration and upscaling)
    print("Beginning restoration and upscaling")
    os.chdir('GFPGAN')
    for adir in aud_dir_names:
        in_pth = str(Path(os.getcwd()).parent.absolute()) + f'/output/vid2Frames/Round1/{adir}/'
        out_pth = str(Path(os.getcwd()).parent.absolute()) + f'/output/GFPGAN/Round1/{adir}/'
        command = f'python inference_gfpgan.py -i {in_pth} -o {out_pth} -v 1.3 -s 4 --bg_upsampler realesrgan'
        try:
            subprocess.call(command, shell=True)
        except subprocess.CalledProcessError:
            print('!!!!!!! Error with GFPGAN command !!!!!!')
            sys.exit()
        print(f'Finished {adir}')
    os.chdir('..')
    print('Completed Restoration Round 1')

    #frames2Vid (Converting frames back to video)
    for adir in aud_dir_names:
        aud_path = glob.glob(f'{audio_super}{adir}/*')[0]
        frames_in_path = f'./output/GFPGAN/Round1/{adir}/restored_imgs/%5d.png'
        vid_out_path = f'./output/frames2Vid/Round1/{adir}.mp4'
        frames2vid(25, aud_path, frames_in_path, vid_out_path)

    #Round1 Printouts
    print("Round 1 Complete!")
    R1end = time.time()
    print("Round1 Elapsed Time:")
    print(R1end - R1start)

  #### Round 2
    print("Beginning Round 2")
    R2start = time.time()

    #FOMM Round 2
    print("First Order Motion Model")
    i=0
    for adir in aud_dir_names:
        ref_video = f'./output/frames2Vid/Round1/{adir}.mp4'
        FOMM_run(face[i], ref_video, generator, kp_detector, adir, Round = "2", relativeTF = False)
        i+=1

    #Vid2Frames R2
    for adir in aud_dir_names:
        frames_out_V2F = f'./output/vid2Frames/Round2/{adir}/'
        vid_path = f'./output/FOMM/Round2/{adir}.mp4'
        os.makedirs(frames_out_V2F, exist_ok=True)
        vid2frames(vid_path, frames_out_V2F)

    #GFPGAN (Restoration and upscaling)
    os.chdir('GFPGAN')
    for adir in aud_dir_names:
        print(f"Final restoration and upscaling of {adir}")
        in_pth = str(Path(os.getcwd()).parent.absolute()) + f'/output/vid2Frames/Round2/{adir}/'
        out_pth = str(Path(os.getcwd()).parent.absolute()) + f'/output/GFPGAN/Round2/{adir}/'
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


    if frame_int is None:
      #Final Frames2Vid
        for adir in aud_dir_names:
            aud_path = glob.glob(f'{audio_super}{adir}/*')[0]
            frames_in_path = f'./output/GFPGAN/Round2/{adir}/restored_imgs/%5d.png'
            vid_out_path = f'./output/frames2Vid/Round2/{adir}.mp4'
            frames2vid(25, aud_path, frames_in_path, vid_out_path)

    else:
        # QVI (Frame interpolation)
        print('Beginning Frame Interpolation.')
        QVIstart = time.time()
        for adir in aud_dir_names:
            os.makedirs(f'./output/QVI/{adir}/', exist_ok=True)
            config = qvi_config(adir, frame_int)
            os.chdir('QVI')
            qvi_main(config)
            os.chdir('..')

            aud_path = glob.glob(f'{audio_super}{adir}/*')[0]
            frames_in_path = f'./output/QVI/{adir}/restored_imgs/*'
            vid_out_path = f'./output/frames2Vid/Round2/{adir}.mp4'
            command = f'ffmpeg -y -r \'{fps}\' -f image2 -pattern_type glob -i \'{frames_in_path}\' -i \'{aud_path}\' -vcodec mpeg4 -b:v 20000k \'{vid_out_path}\''
            subprocess.call(command, shell=True)

        print('Frame Interpolation Complete!')
        QVIend = time.time()
        print("QVI Elapsed Time")
        print(QVIend - QVIstart)

    #Copying to final vids folder
    for adir in aud_dir_names:
        src = f'./output/frames2Vid/Round2/{adir}.mp4'
        final_vids = f'./output/finalVidsOut/{adir}.mp4'
        shutil.copyfile(src, final_vids)

    #Copying final video to save_path
    if save_path != None:
        for adir in aud_dir_names:
            src = f'./output/finalVidsOut/{adir}.mp4'
            shutil.copyfile(src, f'{save_path}{adir}.mp4')

    print('Complete!')
    print('Check ./LIHQ/output/finalVidsOut and your save_path if one was set.')
    