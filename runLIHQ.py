def runModels(refVid, refImg):
  
  #A/V Set up
  auddirnames = get_auddirnames()
  combine_audiofiles()
  
  #FOMM
  FOMM_chop_refvid(refVid, refVidOffset = 0)
  FOMM_loop(refImg, relativeTF=True)

  #Wav2Lip
  Wav2Lip_loop()

  #Vid 2 Frames
  global vidFiles
  vidFiles = glob.glob('/content/Wav2Lip/output/*')
  for vidPath in vidFiles:
    framesOutV2F = '/content/vid2Frames/Round1/' + ntpath.basename(vidPath)[:-4] + '/'
    %mkdir '{framesOutV2F}'
    vid2frames(vidPath, framesOutV2F)

  #ESRGAN
  ESRGAN_loop(vidFiles)

  #GPEN
  #Play around with SR=True/False
  GPEN_loop("Round1")

  #franme2Vid
  Frame2VidR1()

  #FOMM Round 2
  for dir in auddirnames:
    refVideo = '/content/GPEN/Out/VidOutR1/' + dir + '.mp4'
    FOMM_run(refImg, refVideo, dir, relativeTF = False)

  #Vid2Frames R2
  for dir in auddirnames:
    vidFiles = glob.glob('/content/first-order-model/output/{}/*'.format(dir))
    for vidPath in vidFiles:
      framesOutV2F = '/content/vid2Frames/Round2/' + ntpath.basename(vidPath)[:-4] + '/'
      %mkdir '{framesOutV2F}'
      vid2frames(vidPath, framesOutV2F)

  #ESRGAN R2
  ESRGAN_round2()

  #GPEN R2
  GPEN_loop("Round2")

  #Final Frames2Vid
  Frame2VidR2()