import os
import ntpath
import glob
import cv2

from LIHQ.GPEN.face_enhancement import FaceEnhancement

#Functions
def GPEN_run(dir, Round):
  framesOutESR = f'./BasicSR/Out/Round{Round}/{dir}'
  filesIn = glob.glob(framesOutESR + '/*')

  for filex in filesIn:
    #Setting directories
    base_num = ntpath.basename(filex)[:-11]
        
    model = {'name':'GPEN-BFR-512', 'size':512, 'channel_multiplier':2, 'narrow':1, 'use_cuda':True}
    
    outdir = f'./GPEN/Out/{Round}/{dir}/'
    os.makedirs(outdir, exist_ok=True)

    faceenhancer = FaceEnhancement(use_sr=False, device='cuda', size=model['size'], model=model['name'], channel_multiplier=model['channel_multiplier'], narrow=model['narrow'])
    
    im = cv2.imread(filex, cv2.IMREAD_COLOR) # BGR

    img, _, _ = faceenhancer.process(im)
    
    im = cv2.resize(im, img.shape[:2][::-1])
    cv2.imwrite(outdir + base_num +'.png', img)
    print(base_num)