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
    os.mkdir(f'/content/first-order-model/input-ref-vid/{dir}')
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
    os.mkdir(f'/content/first-order-model/output/{dir}')
    vidFiles = glob.glob('/content/first-order-model/input-ref-vid/{}/{}.mp4'.format(dir, dir))
    for vidPath in vidFiles:
      FOMM_run(refImg, vidPath, dir, relativeTF)