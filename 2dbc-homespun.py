import numpy as np
import cv2
import matplotlib.pyplot as plt

# Define canvas size
size = 1024
height = size+1
width = size+1

# Get subdivision sizes
n = 2**np.floor(np.log(height)/np.log(2))  # Greatest power of 2 less than or equal to canvas
# Extract the exponent
n = int(np.log(n)/np.log(2))
boxes = list(2**np.arange(n-1, 1, -1))
boxes.reverse()
boxcounts = {box: 0 for box in boxes}  # Dictionary to store counts for each box size

# Load image
file = 's2_d1.5.png'
canvas = cv2.imread('sample_fractals/' + file, 3)
canvas = cv2.resize(canvas, dsize=[width, height], interpolation=cv2.INTER_AREA)
# Specify a threshold 0-255
threshold = 128
# Make all pixels 0-1
#binarized = np.pad(canvas, pad_width=1, mode='constant', constant_values=1)
binarized = canvas / 255


# 2d box counting function
def boxcount2d(sizes, save=False, visualize=False):
    for s in sizes:

        # Reset box counter
        bc = 0

        # Make list of box corners and clear corner index
        corners = []
        c = 1

        # Make list of corners
        for y in range(0, height, int(height/s)):
            for x in range(0, height, int(height/s)):
                corners.append([x, y])

        # Detect edges and draw box
        #for row in range(s):
            #for pt in range(s):
        for c in range(1, len(corners)-s):
            boxval = np.mean(binarized[corners[c-1][1]:corners[c+s][1], corners[c-1][0]:corners[c][0]])
            if 0.0 < boxval < 1.0:
                cv2.rectangle(canvas, corners[c], corners[c+s], (0, 0, 255), 1)
                bc += 1
            #c += 1
        boxcounts[s] = bc
        print(len(corners))
        print(c+s)
        if visualize:
            cv2.imshow('img', canvas)
            cv2.waitKey(16)

        if save:
            cv2.imwrite('output/' + file, canvas)

    return canvas, boxcounts


image, counts = boxcount2d(boxes, visualize=True)
print(counts)

# Get slope and plot
x = [x for x in counts.keys()]
y = [y for y in counts.values()]

m, b = np.polyfit(np.log(x), np.log(y), 1)
print(m)

#plt.xscale("log")
#plt.yscale("log")

plt.plot(np.log(x), m*np.log(x)+b, label=f"{round(m, 2)}")
plt.scatter(np.log(x), np.log(y))

leg = plt.legend()
plt.show()
