import numpy as np
import cv2
from os import listdir
from os.path import isfile, join

mypath = '/solid/'

# Import Images
filenames = [f for f in listdir(mypath) if isfile(join(mypath, f))]
#images = []
edges = []

for i in len(filenames):
    image = (cv2.imread(mypath + filenames[i], 1))
    canny = cv2.Canny(image, 100, 200)
    canny = cv2.bitwise_not(canny)
    cv2.imwrite(f'/edges/', {filenames[i]}, canny)

