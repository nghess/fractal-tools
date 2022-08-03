import os
import cv2
from os import listdir
from os.path import isfile, join

os.chdir('D:/ImageJ/tutorial')
mypath = 'D:/ImageJ/tutorial/solid/'

# Import Images
filenames = sorted([f for f in listdir(mypath) if isfile(join(mypath, f))])

for i in range(len(filenames)):
    image = cv2.imread(mypath + filenames[i], 0)  # Load grayscale
    canny = cv2.Canny(image, 100, 200)  # Edge Extract
    canny = cv2.bitwise_not(canny)  # Invert
    cv2.imwrite(f'D:/ImageJ/tutorial/edges/{filenames[i]}', canny)
