import os
import sys
import glob
import imageio
import numpy as np
import librosa
from skimage.transform import resize
from IPython.display import HTML
from demo import make_animation
from skimage import img_as_ubyte
import ntpath
import warnings
import moviepy.editor as mpe
from moviepy.video.io.VideoFileClip import VideoFileClip


def FOMM_chop_refvid(auddirnames, refVid, refVidOffset = [0]):
  # Create chopped ref vids into segments the same length as the audio
  i = 0
  
  if len(refVidOffset) < len(auddirnames):
    offset = np.array(refVidOffset)
    offsets = np.pad(offset, (0, len(auddirnames)-len(refVidOffset)), 'constant')
  
  for dir in auddirnames:
    os.mkdir(f'./first_order_model/input-ref-vid/{dir}')
    audio = glob.glob(f'./input/audio/{dir}/*')[0]
    audioLength = librosa.get_duration(filename = audio)

    output_video_path = f'./first_order_model/input-ref-vid/{dir}/{dir}.mp4'
    with VideoFileClip(refVid) as video:
      totalAudioLength = offsets[i] + audioLength
      if video.duration < totalAudioLength:
        sys.exit('Reference video is shorter than audio. You can:',
                 'Chop audio to multiple folders, reduce video offset,',
                 'use a longer reference video, use shorter audio.')
      
      new = video.subclip(offsets[i], offsets[i] + audioLength)
      new.write_videofile(output_video_path, audio_codec='aac')
      i += 1

#Output is in first-order-model folder (output or output-with-sound)
def FOMM_run(SourceImgPath, SourceVidPath, generator, kp_detector, dir, relativeTF = True):
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

  #save resulting video
  FOMMoutPath = f'./first_order_model/output/{dir}/' + ntpath.basename(SourceVidPath)[:-4] + '.mp4'
  imageio.mimsave(FOMMoutPath, [img_as_ubyte(frame) for frame in predictions], fps=fps)