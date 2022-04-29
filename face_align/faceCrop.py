from drive import open_url
from shape_predictor import align_face
import dlib
import torchvision

########################################
## Stolen and Modified from PULSE face restoration
## https://github.com/adamian98/pulse
########################################

def crop_face(filename):
  #downloading model weights
  f=open_url("https://drive.google.com/uc?id=1huhv8PYpNNKbGCLOaYUjOgR1pY5pmbJx", cache_dir="cache", return_path=True)
  predictor = dlib.shape_predictor(f)

  toPIL = torchvision.transforms.ToPILImage()
  toTensor = torchvision.transforms.ToTensor()

  images = []
  for face in align_face(filename,predictor):
    face = toPIL(D(toTensor(face).unsqueeze(0).cuda()).cpu().detach().clamp(0,1)[0])
    images.append(face)
    face.save('/content/Faces/Cropped/output.png')

  if(len(images)==0):
    raise Exception("No faces found. Try again with a different image.")