# LIHQ
#### Long-Inference, High Quality Synthetic Speaker

LIHQ is not a new architecture, it is an application that utilizes several open source deep learning models to generate an artifical speaker of your own design. It was built to be run in google colab to take advantage of free/ cheap GPU with basically zero setup and designed to be as user friendly as I could make it. It was not created for deepfake purposes but you can give it a try if you want (See LIHQ Examples colab and secondary Demo Video). You will find that some voices or face images will not give your desired output and will take a little trial and error to get right. LIHQ really works best with a stylegan2 face and a simple narrator voice. Creating a simple speaker video with a styleGAN2 face and a simple TorToiSe voice is pretty straightforward and often produces good output.

### Update:  
Looks like Bark (https://github.com/suno-ai/bark) is now the open source SOTA for speech generation. Try it out instead of TorToiSe.

![LIHQ Examples](./docs/demo_gif.gif)

## How it works
#### Steps you need to take:

1) Run Setup
2) Create/ Upload Audio (TorToiSe, VITS)
3) Upload Speaker Face Image (StyleGAN2 preferred)
4) (Optional) Add reference video
5) (Optional) Replace Background

#### Steps the program takes:

6) Face/ Head Motion Transfer (FOMM)
7) Mouth Motion Generation (Wav2Lip)
8) Upscale and Restoration (GFPGAN)
9) Second Run Through (FOMM and GFPGAN)
10) (Optional) Frame Interpolation (QVI) (Noticable improvement but long inference)
11) (Optional) Background Matting (MODNet)

Pick out an image of a face that is forward-facing with a closed mouth (https://github.com/NVlabs/stylegan2) and upload or create audio of anyone you want using TorToiSe (built into the LIHQ colab) https://github.com/neonbjb/tortoise-tts

LIHQ will first transfer head and eye movement from my default reference video to your face image using a First Order Motion Model. Wav2Lip will then create mouth movement from your audio and paste it onto the FOMM output. Since the output is a very low resolution (256x256) we need to run through a face restoration & super resolution model. Repeating this process a second time will make the video look even better. And if you want it to be the highest quality, you can add frame interpolation at the end to increase the fps.

## Demo Video
[![LIHQ Demo Video](https://img.youtube.com/vi/PXTiR_S3UuY/0.jpg)](https://www.youtube.com/watch?v=PXTiR_S3UuY)

Above is the primary LIHQ demo video, demonstrating the software being used as intended. Deepfakes are possible in LIHQ as well but they take a bit more work. Check out some examples in the [Deepfake Example Video](https://www.youtube.com/watch?v=nPAV-jpTzqI).

## Colabs

The google colabs have a lot of information, tips and tricks that I will not be putting in the README. LIHQ has a cell for every feature you might want or need with an explanation of everything. It is lengthy and a little cluttered but I recommend reading through everything. LIHQ Examples has four different examples, trimmed down to only what you need for each example.

[LIHQ](https://colab.research.google.com/drive/1fKZl59AVDR4oGvlhVXdyCUGuozpnbIgQ?usp=sharing)

[LIHQ Examples](https://colab.research.google.com/drive/1rIgl8J-EMJ4BcSPjKNsVk8BdproD98WW?usp=sharing)

## Possible Future Work
- Create more reference videos. Lip movement seems to work best at ref_vid_offset = 0 and I don't think this is coincidence. I think it has to do with the FOMM  movement transfer. I may create more reference videos of shorter length, and with varying emotions.
- Make wav2LIP, FOMM optional. If you want to use reference video with correct lip movement, or a speaker video that already contains target speaker.
- Add randomizer to ref vid offset
- Expand post-processing capabilities
- Revise for local use. I'm not sure what it would take. It's probably 95% there, I just simply haven't tried outside of colab.
