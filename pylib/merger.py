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

        self.save_diff = False

    def calc_mean(self):
        self.logger.debug("Calculating mean.")
        self.mean_image = sum(self.input.get_frames()) / self.input.number_images

    def calc_diff(self, frame):
        return self.mean_image - frame

    def calc_metric(self, diff):
        # change intensity to increase emphasis of outliers
        intensity = 500
        metric = 1 + intensity * np.sqrt(np.sum(np.square(diff/255), axis=-1))
        metric = cv2.GaussianBlur(metric, (5,5), 0)
        return metric

    def merge_frame(self, frame, index):
        diff = self.calc_diff(frame)
        if self.save_diff:
            self.save_image(diff, os.path.join("out", "diff_{:03}.png".format(index)))

        metric = self.calc_metric(diff)

        self.sum_weights += metric

        metric = metric.reshape((self.input.shape[0], self.input.shape[1], 1))

        weighted_frame = frame * metric
        self.merged_image += weighted_frame

    def merge_all(self):

        if self.mean_image is None:
            self.calc_mean()

        self.logger.debug("Calculating merged.")

        self.sum_weights = np.ndarray(shape=(self.input.shape[0], self.input.shape[1]),
                                      dtype=np.float)
        self.merged_image = np.ndarray(shape=self.input.shape, dtype=np.float)

        index = 0
        for frame in self.input.get_frames():
            self.merge_frame(frame, index)
            index += 1

        self.merged_image = self.merged_image / self.sum_weights.reshape((self.input.shape[0], self.input.shape[1], 1))

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

class DefaultMerger(Merger):

    def __init__(self, inpt):
        super().__init__(inpt)

class CutoffMerger(Merger):

    def __init__(self, inpt):
        super().__init__(inpt)
        self.save_diff = True

    def calc_mean(self):
        for frame in self.input.get_frames():
            self.mean_image = frame
            break

    def calc_metric(self, diff):
        metric = np.sqrt(np.sum(np.square(diff/255), axis=-1))
        metric = np.piecewise(metric, [metric < 0.1, metric >= 0.1], [0.1/self.input.number_images, 1])
        return metric