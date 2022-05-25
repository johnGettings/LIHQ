import os
import sys
import glob
import imageio
import numpy as np
import librosa
from skimage.transform import resize
from demo import make_animation
from skimage import img_as_ubyte
import warnings
from moviepy.video.io.VideoFileClip import VideoFileClip


def FOMM_chop_refvid(aud_dir_names, ref_vid, audio_super, ref_vid_offset):
    # Create chopped ref vids into segments the same length as the audio
    i = 0

    offset = np.array(ref_vid_offset)
    if len(ref_vid_offset) < len(aud_dir_names):
        offset = np.pad(offset, (0, len(aud_dir_names)-len(ref_vid_offset)), 'constant')

    for adir in aud_dir_names:
        os.makedirs(f'./first_order_model/input-ref-vid/{adir}', exist_ok=True)
        audio = glob.glob(f'{audio_super}{adir}/*')[0]
        audio_length = librosa.get_duration(filename = audio)

        output_video_path = f'./first_order_model/input-ref-vid/{adir}/{adir}.mp4'
        with VideoFileClip(ref_vid) as video:
            total_audio_length = offset[i] + audio_length
            if video.duration < total_audio_length:
                sys.exit('Reference video is shorter than audio. You can:',
                        'Chop audio to multiple folders, reduce video offset,',
                        'use a longer reference video, use shorter audio.')

            new = video.subclip(offset[i], offset[i] + audio_length)
            new.write_videofile(output_video_path, audio_codec='aac')
            i += 1

#Output is in first-order-model folder (output or output-with-sound)
def FOMM_run(source_img_path, source_vid_path, generator, kp_detector, adir, Round, relativeTF = True):
    warnings.filterwarnings("ignore")

    source_image = imageio.imread(source_img_path)
    reader = imageio.get_reader(source_vid_path)

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

    #save resulting video
    FOMM_out_path = f'./output/FOMM/Round{Round}/{adir}.mp4'
    imageio.mimsave(FOMM_out_path, [img_as_ubyte(frame) for frame in predictions], fps=fps)
    