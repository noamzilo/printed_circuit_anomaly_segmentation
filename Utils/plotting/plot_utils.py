import numpy as np
from matplotlib.pyplot import figure, imshow
import matplotlib.pyplot as plt
from Utils.ConfigProvider import ConfigProvider


class Plotter(object):
    config = ConfigProvider.config()
    is_plotting = config.misc.is_plotting

    @staticmethod
    def show_color_diff(im1, im2, title):
        if not Plotter.is_plotting:
            return
        to_show = Plotter.get_color_diff_image(im1, im2)
        figure()
        plt.title(title)
        imshow(to_show)

    @staticmethod
    def get_color_diff_image(im1, im2):
        to_show = np.ones((im1.shape[0], im1.shape[1], 3))
        to_show[:, :, 0] = im1
        to_show[:, :, 1] = im2
        to_show[:, :, 2] = im1
        to_show = to_show.astype('uint8')
        return to_show


    @staticmethod
    def show_color_diff_threeway(im1, im2, im3, title):
        to_show = np.ones((im1.shape[0], im1.shape[1], 3))
        to_show[:, :, 0] = im1
        to_show[:, :, 1] = im2
        to_show[:, :, 2] = im3
        to_show = to_show.astype('uint8')
        figure()
        plt.title(title)
        imshow(to_show)
        plt.show()
        return to_show

    @staticmethod
    def plot_image(im, title=""):
        if not Plotter.is_plotting:
            return
        figure()
        plt.title(title)
        imshow(im, cmap='gray')

    @staticmethod
    def plot_image_3d(im):
        if not Plotter.is_plotting:
            return
        xx, yy = np.mgrid[0:im.shape[0], 0:im.shape[1]]
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.plot_surface(xx, yy, im, rstride=1, cstride=1, cmap=plt.cm.gray, linewidth=0)


show_color_diff = Plotter.show_color_diff
plot_image = Plotter.plot_image
plot_image_3d = Plotter.plot_image_3d
show_color_diff_threeway = Plotter.show_color_diff_threeway
get_color_diff_image = Plotter.get_color_diff_image
