import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
import IPython
import os

from utils.audio import load_audio, get_voices

def tort(tts, text, voice, preset):
  voices = get_voices()
  cond_paths = voices[voice]
  conds = []
  for cond_path in cond_paths:
      c = load_audio(cond_path, 22050)
      conds.append(c)

  gen = tts.tts_with_preset(text, conds, preset)
  torchaudio.save('generated.wav', gen.squeeze(0).cpu(), 24000)
  IPython.display.Audio('generated.wav')