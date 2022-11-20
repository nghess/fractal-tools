import numpy as np
import cv2

height = 1024
width = 1024

canvas = np.zeros((height, width, 3), dtype=np.uint8)

# Minimal dimension of image
p = height

# Greatest power of 2 less than or equal to p
n = 2**np.floor(np.log(p)/np.log(2))

# Extract the exponent
n = int(np.log(n)/np.log(2))

# Build successive box sizes (from 2**n down to 2**1)
sizes = 2**np.arange(n, 1, -1)

count = 0
multiplier = 0
sz_multis = []
k = 0

for x in sizes:
    k = x
    while k <= p:
        multiplier += 1
        k = k + x
    sz_multis.append(multiplier)
    multiplier = 0


# Create corners list
grid = [[0, 0]]
x1 = 0

for x in range(sz_multis[2]):
    y1 = 0
    for y in range(sz_multis[2]):
        y1 = y1 + sizes[2]
        grid.append([x1, y1])
    x1 = x1 + sizes[2]

# Draw boxes
brightness = 255-int(255/len(grid))
for b in range(len(grid)-sz_multis[2]):
    color = int(255/(b+1))
    canvas = cv2.rectangle(canvas, grid[b], grid[b+sz_multis[2]], (color, color, 0), -1)


# Box sizes and packing
#print(sizes)
#print(sz_multis)

print(grid)

cv2.imshow('Grid', canvas)
cv2.waitKey(100000000)