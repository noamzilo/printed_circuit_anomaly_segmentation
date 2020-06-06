import numpy as np
from matplotlib.pyplot import figure, imshow
import matplotlib.pyplot as plt


def show_color_diff(im1, im2, title):
    to_show = np.ones((im1.shape[0], im1.shape[1], 3))
    to_show[:, :, 0] = im1
    to_show[:, :, 1] = im2
    to_show[:, :, 2] = im1
    to_show = to_show.astype('uint8')
    figure()
    plt.title("title")
    imshow(to_show)


def plot_image(im, title=""):
    figure()
    plt.title(title)
    imshow(im, cmap='gray')


def plot_image_3d(im):
    xx, yy = np.mgrid[0:im.shape[0], 0:im.shape[1]]
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(xx, yy, im, rstride=1, cstride=1, cmap=plt.cm.gray, linewidth=0)
