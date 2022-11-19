import numpy as np
import cv2


#Z = np.zeros([64,64], dtype='int16')
Z = cv2.imread('sample_fractals/s2_d1.0.png', 0)
Z = cv2.resize(Z, dsize=[100, 100], interpolation=cv2.INTER_AREA)

k = 4#

S = np.add.reduceat(
    np.add.reduceat(Z, np.arange(0, Z.shape[0], k), axis=0),
    np.arange(0, Z.shape[1], k), axis=1)

print(list(S))

cv2.imshow('img', Z)
cv2.waitKey(0)


# try different approach than reduce at. it is cool but it is too abstract.
