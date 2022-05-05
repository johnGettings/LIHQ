import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
import IPython
import os

from utils.audio import load_audio, get_voices

def tortoise_run(tts, text, voice, preset):
  voices = get_voices()
  cond_paths = voices[voice]
  conds = []
  for cond_path in cond_paths:
      c = load_audio(cond_path, 22050)
      conds.append(c)

  gen = tts.tts_with_preset(text, conds, preset)
  return gen
  
  
def tortoise_combo_run(tts, text, voice1, voice2, preset):
  conds = []
  voices = get_voices()
  for v in [voice1, voice2]:
    cond_paths = voices[v]
    for cond_path in cond_paths:
        c = load_audio(cond_path, 22050)
        conds.append(c)

  gen = tts.tts_with_preset(text, conds, preset)
  return gen