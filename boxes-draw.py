import numpy as np
import cv2

size = 1024

height = size+1
width = size+1
subdiv = 128

# Load image
canvas = cv2.imread('sample_fractals/s2_d1.2.png', 3)
canvas = cv2.resize(canvas, dsize=[height, width], interpolation=cv2.INTER_AREA)
# Specify a threshold 0-255
threshold = 128
# make all pixels < threshold black
binarized = 1.0 * (canvas > threshold)

# Greatest power of 2 less than or equal to p
n = 2**np.floor(np.log(height)/np.log(2))
# Extract the exponent
n = int(np.log(n)/np.log(2))

# Make list of box corners
corners = []
for y in range(0, height, int(height/subdiv)):
    for x in range(0, height, int(height/subdiv)):
        corners.append([x, y])

# Detect edges and draw box
for c in range(1, len(corners)-subdiv):
    boxval = np.mean(binarized[corners[c-1][1]:corners[c+subdiv][1], corners[c-1][0]:corners[c][0]])
    if 0.0 < boxval < 1.0:
        cv2.rectangle(canvas, corners[c], corners[c+subdiv], (0, 0, 255), 1)


cv2.imshow('img', canvas)
cv2.waitKey(1000000)