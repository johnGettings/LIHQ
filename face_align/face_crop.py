from pathlib import Path
from google.colab import files
from io import BytesIO
from matplotlib import pyplot as plt
from PIL import Image
import torchvision
from IPython import display
import numpy as np
from shape_predictor import align_face
from drive import open_url
import dlib
from mpl_toolkits.axes_grid1 import ImageGrid
from bicubic import BicubicDownSample

######################################
# Stolen from PULSE face depixelization
# https://github.com/adamian98/pulse

def crop_face(filename, outfile):
  #downloading model weights
  f=open_url("https://drive.google.com/uc?id=1huhv8PYpNNKbGCLOaYUjOgR1pY5pmbJx", cache_dir="cache", return_path=True)
  predictor = dlib.shape_predictor(f)

  toPIL = torchvision.transforms.ToPILImage()
  toTensor = torchvision.transforms.ToTensor()

  images = []
  for face in align_face(filename,predictor):
    face = toPIL(toTensor(face).unsqueeze(0).cuda().cpu().detach().clamp(0,1)[0])
    images.append(face)
    face.save(outfile)

  if(len(images)==0): raise Exception("No faces found. Try again with a different image.")