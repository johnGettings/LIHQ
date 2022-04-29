import glob
import wave

#Get a list of all subfolders in audio directory
def get_auddirnames():
  audfolders = sorted(glob.glob("/content/audio/*/"))
  global auddirnames
  auddirnames = []
  for ff in audfolders:
    auddirnames.append(os.path.basename(os.path.dirname(ff)))
  return auddirnames

#audiofiles should be in numerical order
#removes everything except combined audio. Coimbined audio name is same as subfolder.
def combine_audiofiles():
  for dir in auddirnames:
    audioFiles = sorted(glob.glob('/content/audio/{}/*'.format(dir)))
    
    if len(audioFiles) > 1:
      outfile = '/content/audio/{}/{}.wav'.format(dir,dir)
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
        os.rm(filez)

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
def Frame2VidR1():
  for dir in auddirnames:
    audPath = glob.glob('/content/audio/{}/*'.format(dir))[0]
    frames = '/content/GPEN/Out/Round1/' + dir + '/*.png'
    outPath = '/content/GPEN/Out/VidOutR1/' + dir + '.mp4'
    os.system(f'ffmpeg -y -r 25 -f image2 -pattern_type glob -i {frames} -i {audPath} -vcodec mpeg4 -b 1500k {outPath}')

def Frame2VidR2():
  for dir in auddirnames:
    audPath = glob.glob('/content/audio/{}/*'.format(dir))[0]
    frames = '/content/GPEN/Out/Round2/' + dir + '/*.png'
    outPath = '/content/FinalVideos/' + faceName + '.mp4'
    os.system(f'ffmpeg -y -r 25 -f image2 -pattern_type glob -i {frames} -i {audPath} -vcodec mpeg4 -b 1500k {outPath}')