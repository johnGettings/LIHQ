import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
import IPython
import os

from tortoise.utils.audio import load_voice, load_voices

def tortoise_run(tts, text, voice, preset):
  voice_samples, conditioning_latents = load_voice(voice)
  gen = tts.tts_with_preset(text, voice_samples=voice_samples,
                          conditioning_latents=conditioning_latents, 
                          preset=preset)

  return gen

def tortoise_combo_run(tts, text, voice1, voice2, preset):
  voice_samples, conditioning_latents = load_voices([voice1, voice2])

  gen = tts.tts_with_preset(text, voice_samples=voice_samples,
                        conditioning_latents=conditioning_latents, 
                        preset=preset)
  return gen