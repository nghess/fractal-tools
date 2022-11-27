import numpy as np
import cv2
import matplotlib.pyplot as plt

# Define canvas size
size = 1024
height = size+1
width = size+1

# Get subdivision sizes
n = 2**np.floor(np.log(height)/np.log(2)) # Greatest power of 2 less than or equal to canvas
# Extract the exponent
n = int(np.log(n)/np.log(2))
sizes = 2**np.arange(n-1, 0, -1)
boxcounts = {size: 0 for size in sizes}

# Load image
file = 's2_d1.8.png'
canvas = cv2.imread('sample_fractals/' + file, 3)
canvas = cv2.resize(canvas, dsize=[height, width], interpolation=cv2.INTER_AREA)
# Specify a threshold 0-255
threshold = 128
# make all pixels < threshold black
binarized = canvas > threshold


def boxcount2d(subdiv):
    for s in subdiv:
        # Reset box counter
        bc = 0
        # Make list of box corners
        corners = []
        for y in range(0, height, int(height/s)):
            for x in range(0, height, int(height/s)):
                corners.append([x, y])

        # Detect edges and draw box
        for c in range(1, len(corners)-s):
            boxval = np.mean(binarized[corners[c-1][1]:corners[c+s][1], corners[c-1][0]:corners[c][0]])
            if 0.0 < boxval < 1.0:
                cv2.rectangle(canvas, corners[c], corners[c+s], (0, 0, 255), 1)
                bc += 1

        boxcounts[s] = bc

        cv2.imshow('img', canvas)
        cv2.waitKey(16)

    return canvas, boxcounts

boxes, boxcounts = boxcount2d(sizes)

#cv2.imwrite('output/' + file, boxes)

print(boxcounts)



# plot
x = [x for x in boxcounts.keys()]
y = [y for y in boxcounts.values()]

m, b = np.polyfit(np.log(x), np.log(y), 1)

#plt.xscale("log")
#plt.yscale("log")

plt.plot(np.log(sizes), m*np.log(sizes)+b, label=f"{round(m, 2)}")

plt.scatter(np.log(x), np.log(y))
leg = plt.legend()

plt.show()