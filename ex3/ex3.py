import numpy as np
import scipy.io     # used for loading MATLAB/Octave *.mat files
from random import sample   # required for obtaining random indices for data visualization
import matplotlib.pyplot as plt # used for visualizing the training data
from matplotlib import cm
from scipy.misc import toimage   # convert matrix to image

datafile = 'ex3data1.mat'

def load_data(file):
    """Load the image data and return it as numpy arrays"""
    mat = scipy.io.loadmat(file)
    X, y = mat['X'], mat['y']
    return X, y

def get_image(X, index):
    """Reshape the row vector at the given index into a 20x20 picture and return it."""
    height, width = 20, 20
    img = X[index].reshape(height, width)
    return img.T

def display_data(X):
    """Display a selection of the feature data (10 x 10 handwritten digits)."""
    width, height = 20, 20
    num_rows, num_cols = 10, 10

    # select num_rows * num_cols random training examples indices
    indices = sample(range(X.shape[0]), num_rows * num_cols)

    # build the big image
    row, col = 0, 0
    big_pic = np.ones( (height * num_rows, width * num_cols) )
    for idx in indices:
        small_pic = get_image(X, idx)
        # reset the column counter if we reach the end of the picture
        # and start a new row
        if col == num_cols:
            col = 0
            row += 1
        y_start, y_end = row * height, row * width + width
        x_start, x_end = col * width, col * width + width
        big_pic[x_start:x_end, y_start:y_end] = small_pic
        col += 1

    # convert the image data to an actual image and show it using a greyscale color map
    figure = plt.figure(figsize=(6,6))
    img = toimage(big_pic)
    plt.imshow(img, cmap= cm.Greys_r)
    plt.show()

X, y = load_data(datafile)
display_data(X)