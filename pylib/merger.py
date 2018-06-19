#!/usr/bin/env python3

from . import log
import numpy as np
import os.path
import os
import cv2

class Merger(object):
    """ This class overlays frames and produces an output image. """
    def __init__(self, input):

        self.input = input

        self.logger = log.setup_logger("merger")

        self.merged_image = None
        self.mean_image = None

    def calc_mean(self):
        self.logger.debug("Calculating mean.")
        self.mean_image = sum(self.input.get_frames()) / self.input.number_images

    def calc_merged(self):
        raise NotImplementedError

    @staticmethod
    def show_image(image):
        cv2.imshow('Image', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def save_image(self, image, path=os.path.join("out", "out.png"), quiet=False):
        _dir = os.path.dirname(path)
        if _dir and not os.path.isdir(_dir):
            try:
                os.mkdir(_dir)
            except:
                self.logger.error("Could not create director '{}'!".format(_dir))
                return False
        if not quiet:
            self.logger.info("Writing merged image to '{}'.".format(path))
        cv2.imwrite(path, image)
        return True

class Merger1(Merger):

    def __init__(self, inpt):
        super().__init__(inpt)

    def calc_merged(self):

        if self.mean_image is None:
            self.calc_mean()

        self.logger.debug("Calculating merged.")

        self.sum_weights = np.ndarray(shape=(self.input.shape[0], self.input.shape[1]),
                                      dtype=np.float)
        self.merged_image = np.ndarray(shape=self.input.shape, dtype=np.float)


        for frame in self.input.get_frames():
            diff = (self.mean_image - frame)/255

            # change intensity to increase emphasis of outliers
            intensity = 500
            metric = 1 + intensity*np.sqrt(np.sum(np.square(diff), axis=-1))

            # blurring in space as an attemptas an attemptas an attempt to reduce noiseee
            metric = cv2.GaussianBlur(metric, (5,5), 0)

            self.sum_weights += metric

            metric = metric.reshape((self.input.shape[0], self.input.shape[1], 1))

            weighted_frame = frame * metric

            self.merged_image += weighted_frame

        self.merged_image = self.merged_image / self.sum_weights.reshape((self.input.shape[0], self.input.shape[1], 1))