# LIHQ
#### Long-Inference, High Quality Synthetic Speaker

This project is fully functional but a work in progress. It is currently designed for google colab because I do not have a local GPU that can compete with what colab offers so I have not tested locally. In the future I will redesign for local applications as well. If there is a lot of demand for a local version I will work on it sooner rather than later. I will continue to tweak parameters to achieve best possible output. If a newer model comes along that outperforms one that LIHQ utilizes, I will replace it. (And please let me know if you think you found something that will work better!) This can be a very collaborative project as well, if the community wishes.

## Inference time
When I say 'long-inference,' I mean it. LIHQ is running up to eight DNNs if you choose to use every feature. This is meant for short hobby projects, not long videos or commercial applications. See below for expected inference times.

## How it works
#### Things you need to do:

1) Setup

2) Create/ Upload Audio

3) Upload Speaker Face Image

4) (Optional) Add reference video

4) (Optional) Upload Background

#### Things the program does:

5) Face/ Head Motion Transfer (FOMM)

6) Mouth Motion Generation (Wav2Lip)

7) Upscale and Restoration (ESRGAN & GPEN)

8) (Optional) but Recommended Second Run Through

9) (Optional) Frame Interpolation

10) (Optional) Greenscreen background

Pick out an image of a face that is forward-facing with a closed mouth and upload or create audio of anyone you want using TorToiSe (built into the LIHQ colab) https://github.com/neonbjb/tortoise-tts

LIHQ will first transfer head and eye movement from my default reference video to your face image using a First Order Motion Model. Wav2Lip will then create mouth movement from your audio and paste it onto the FOMM output. Since the output is a very low resolution we need to run through a face restoration & super resolution model. Repeating this process a second time will make the video look even better. And if you want it to be the highest quality, you can add frame interpolation to increase the fps.

## Parameters

The only two critical parameters is your audio and your face. If you wish, you can add a background or greenscreen. You may choose the output quality and framerate if you want a faster inference time. Each of the individual models have various parameters you can play around with as well, but you would need to change those in the procedures.py file. I have played with them a lot and the ones I recommend.

## Demo Videos

Link to main demo

Link to demonstration of various options

## Colabs

LIHQ

LIHQ Clean
