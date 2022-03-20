import gist
import numpy as np
import cv2

img = ... cv2.imread('inpainting-in-medical-imaging/images/test.png')
descriptor = gist.extract(img)