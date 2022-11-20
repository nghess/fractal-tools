import numpy as np
import cv2



def fractal_dimension_2d(Z, threshold=0.9):

    # Only for 2d image
    assert(len(Z.shape) == 2)


    # From https://github.com/rougier/numpy-100 (#87)
    def boxcount(Z, k):
        S = np.add.reduceat(
            np.add.reduceat(Z, np.arange(0, Z.shape[0], k), axis=0),
            np.arange(0, Z.shape[1], k), axis=1)

        # We count non-empty (0) and non-full boxes (k*k)
        return len(np.where((S > 0) & (S < k*k))[0]), np.where((S > 0) & (S < k*k))

    # Transform Z into a binary array
    Z = (Z < threshold)

    # Minimal dimension of image
    p = min(Z.shape)

    # Greatest power of 2 less than or equal to p
    n = 2**np.floor(np.log(p)/np.log(2))

    # Extract the exponent
    n = int(np.log(n)/np.log(2))

    # Build successive box sizes (from 2**n down to 2**1)
    sizes = 2**np.arange(n, 1, -1)

    # Actual box counting with decreasing size
    counts = []
    for size in sizes:
        count, hits = boxcount(Z, size)
        counts.append(count)

    # Fit the successive log(sizes) with log (counts)
    m, b = np.polyfit(np.log(sizes), np.log(counts), 1)
    return -m, b, sizes, counts, hits

#load fractal

img = cv2.imread('sample_fractals/s2_d1.2.png', 0)

slope, intercept, sizes, counts, hits = fractal_dimension_2d(img)

print(sizes)
print(counts)
print(len(hits[0]))


cv2.imshow('fractal', img)
cv2.waitKey(1000000)


