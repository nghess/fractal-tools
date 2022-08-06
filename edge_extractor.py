import os
import cv2
from os import listdir
from os.path import isfile, join


def edge_extract(i_dir, o_dir, lo=100, hi=200):

    # Set up paths
    cur_dir = os.path.dirname(__file__)
    input_dir = f'{cur_dir}/{i_dir}/'
    output_dir = f'{cur_dir}/{o_dir}/'

    # Import Images
    filenames = sorted([f for f in listdir(input_dir) if isfile(join(input_dir, f))])

    # Check if output dir exists
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # Extract edges and save each file
    for i in range(len(filenames)):
        image = cv2.imread(input_dir + filenames[i], 0)  # Load grayscale
        canny = cv2.Canny(image, lo, hi)  # Edge Extract
        canny = cv2.bitwise_not(canny)  # Invert
        cv2.imwrite(f'{output_dir}{filenames[i]}', canny)


edge_extract('sample_fractals', 'test')
