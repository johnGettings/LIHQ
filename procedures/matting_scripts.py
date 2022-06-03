import numpy as np
import os
import sys
from PIL import Image
import cv2
import mimetypes
from moviepy.editor import *


def image_matting(background_image, face_image, mask_folder, output_folder):

    #Opening Images
    image_name = face_image
    matte_name = os.path.basename(image_name).split('.')[0] + '.png'
    background = Image.open(background_image) #Set background image
    image = Image.open(face_image)
    matte = Image.open(os.path.join(mask_folder, matte_name))

    #Reshaping background as needed
    if background.size != image.size:
        background = background.resize((image.size[0], image.size[1]))

    #Matte transformaiton
    image = np.asarray(image)
    matte = np.repeat(np.asarray(matte)[:, :, None], 3, axis=2) / 255
    foreground = image * matte + background * (1 - matte)
    final = Image.fromarray(np.uint8(foreground))
    final.save(output_folder + '/matte_' + matte_name)
  
def matte_preview(speaker_vid, background, bg_resize, spkr_resize, offset):

    #Video or image
    if mimetypes.guess_type(background)[0].startswith('video'):
        video_background = True
    else:
        video_background = False

    #First frame of speaker
    vidcap = cv2.VideoCapture(speaker_vid)
    _, speaker = vidcap.read()
    if speaker is None:
      sys.exit('Bad file/path for speaker vid.')

    #Frame of background
    if video_background:
        vidcap = cv2.VideoCapture(background)
        _, new_background = vidcap.read()
    else:
        new_background = cv2.imread(background)
    if speaker is None:
        sys.exit('Bad file/path for background.')
        
    #Dimensions
    print('This is just a preview! Run video matting cell below once image is configured correctly.')
    print(f'Original size of Background: {new_background.shape[0]} x {new_background.shape[1]}')
    print(f'Original size of Speaker Video: {speaker.shape[0]} x {speaker.shape[1]}')

    #Resize
    if bg_resize:
        new_background = cv2.resize(new_background, bg_resize)
    if spkr_resize:
        speaker = cv2.resize(speaker, spkr_resize)

    cv2.imwrite('./output/postprocessing/input/preview.png', speaker)
    cv2.imwrite('./output/postprocessing/background/preview.png', new_background)

    # Dimensions
    spkr_w = speaker.shape[1]
    spkr_h = speaker.shape[0]
    bg_w = new_background.shape[1]
    bg_h = new_background.shape[0]
    cntr_btm = [round(bg_h - spkr_h), round(bg_w/2 - spkr_w/2)]
    cntr_cntr = [round(bg_h/2 - spkr_h/2), round(bg_w/2 - spkr_w/2)]

    print(f'Offset Required for Center Bottom: {cntr_btm}')
    print(f'Offset Required for Center Center: {cntr_cntr}')
    
def matte_combine(offset, rotation):
    # Combining mask, source, and backgorund
    input_folder = './output/postprocessing/input'
    background_folder = './output/postprocessing/background'
    output_folder = './output/postprocessing/masks'
    image_names = os.listdir(input_folder)
    background_names = os.listdir(background_folder)
    
    i=0
    for image_name in image_names:
        print(f'{i} of {len(image_names)}')
        base_name = os.path.basename(image_name).split('.')[0] + '.png'
        
        if len(background_names) == 1:
          background_image = f'./output/postprocessing/background/{background_names[0]}'
        else:
          background_image = f'./output/postprocessing/background/{base_name}'

        background = Image.open(background_image) #Set background image

        #Pasting onto black bg
        black_matte = np.full_like(background, (0,0,0))
        black_spkr = np.full_like(background, (0,0,0))

        matte_cv = cv2.imread(os.path.join(output_folder, base_name))
        spkr_cv = cv2.imread(os.path.join(input_folder, base_name))

        height, width = matte_cv.shape[:2]
        center = (width/2, height/2)
        rotate_matrix = cv2.getRotationMatrix2D(center=center, angle=rotation, scale=1)
        matte_cv = cv2.warpAffine(src=matte_cv, M=rotate_matrix, dsize=(width, height))
        spkr_cv = cv2.warpAffine(src=spkr_cv, M=rotate_matrix, dsize=(width, height))

        black_matte[offset[0]:offset[0] + matte_cv.shape[0], offset[1]:offset[1] + matte_cv.shape[1]] = matte_cv
        black_spkr[offset[0]:offset[0] + spkr_cv.shape[0], offset[1]:offset[1] + spkr_cv.shape[1]] = spkr_cv

        image = cv2.cvtColor(black_spkr, cv2.COLOR_BGR2RGB) 
        matte = black_matte / 255
        foreground = image * matte + background * (1 - matte)
        final = Image.fromarray(np.uint8(foreground))

        final.save(f'./output/postprocessing/output/{base_name}' )

        i += 1

def matte_video(speaker_vid, background, bg_resize, spkr_resize):

    #Video or image
    if mimetypes.guess_type(background)[0].startswith('video'):
        video_background = True
    else:
        video_background = False

    #Initializing speaker
    vidcap_spkr = cv2.VideoCapture(speaker_vid)
    fps_spkr = vidcap_spkr.get(cv2.CAP_PROP_FPS)
    frames_spkr = int(vidcap_spkr.get(cv2.CAP_PROP_FRAME_COUNT))
    dur_spkr = frames_spkr/fps_spkr

    #Initializing background
    if video_background:
        vidcap_bg = cv2.VideoCapture(background)
        _, frame = vidcap_bg.read()
        fps_bg = vidcap_bg.get(cv2.CAP_PROP_FPS)
        frames_bg = int(vidcap_bg.get(cv2.CAP_PROP_FRAME_COUNT))
        dur_bg = frames_bg/fps_bg
        if dur_bg < dur_spkr:
            sys.exit('Error. Background video is shorter than speaker video.')
        if fps_bg != fps_spkr:
            if fps_bg < fps_spkr:
                print('FPS of background video is less than that of speaker video. FPS of speaker is being reduced.')
                clip = VideoFileClip(speaker_vid)
                clip.write_videofile('./output/postprocessing/tempspkr.mp4', fps=fps_bg)
                
                vidcap_spkr = cv2.VideoCapture('./output/postprocessing/tempspkr.mp4')
                frames_spkr = int(vidcap_spkr.get(cv2.CAP_PROP_FRAME_COUNT))
            else:
                print('FPS of speaker video is less than that of background video. FPS of background is being reduced.')
                clip = VideoFileClip(background)
                clip.write_videofile('./output/postprocessing/tempbg.mp4', fps=fps_spkr)
                vidcap_bg = cv2.VideoCapture('./output/postprocessing/tempbg.mp4')
        if bg_resize == False:
            bg_resize = (frame.shape[1], frame.shape[0])
    else:
        imgcap_bg = cv2.imread(background) 
        if bg_resize == False:
            bg_resize = (imgcap_bg.shape[1], imgcap_bg.shape[0])
    
    fps = vidcap_spkr.get(cv2.CAP_PROP_FPS)

    #Height and width must be even number
    if bg_resize[0] & 1:
        bg_resize = (bg_resize[0]-1, bg_resize[1])
    if bg_resize[1] & 1:
        bg_resize = (bg_resize[0], bg_resize[1]-1)

    #Saving frames of speaker. Pasting on top of white image the size of bg
    success,image = vidcap_spkr.read()
    frame = 1
    while success:
        if spkr_resize:
            image = cv2.resize(image, spkr_resize)
        cv2.imwrite('./output/postprocessing/input/' + str(frame).zfill(5) + '.png', image)
        success,image = vidcap_spkr.read()
        frame += 1

    #Saving frames of background
    if video_background:
        success,image = vidcap_bg.read()
        frame = 1
        while success:
            image = cv2.resize(image, bg_resize)
            cv2.imwrite('./output/postprocessing/background/' + str(frame).zfill(5) + '.png', image)
            success,image = vidcap_bg.read()
            frame += 1
    else:
        frame = 1
        if bg_resize:
                imgcap_bg = cv2.resize(imgcap_bg, bg_resize)
        while frame <= frames_spkr:
            cv2.imwrite('./output/postprocessing/background/' + str(frame).zfill(5) + '.png', imgcap_bg)
            frame += 1
    
    return fps
