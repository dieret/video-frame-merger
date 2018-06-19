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
        self.save_mean = True

    def calc_mean(self):
        self.logger.debug("Calculating mean.")
        self.mean_image = self._calc_mean()
        if self.save_mean:
            self.save_image(self.mean_image, "out/mean.png")

    def _calc_mean(self):
        return sum(self.input.get_frames()) / self.input.number_images

    def _calc_diff(self, frame):
        if self.mean_image is None:
            self.calc_mean()
        return self.mean_image - frame

    def _calc_metric(self, diff):
        # change intensity to increase emphasis of outliers
        intensity = 500
        metric = 1 + intensity * np.sqrt(np.sum(np.square(diff/255), axis=-1))
        metric = cv2.GaussianBlur(metric, (5,5), 0)
        return metric

    def merge_frame(self, frame, index):
        diff = self._calc_diff(frame)
        if self.save_diff:
            self.save_image(diff, os.path.join("out", "diff_{:03}.png".format(index)))

        metric = self._calc_metric(diff)

        self.sum_weights += metric

        metric = metric.reshape((self.input.shape[0], self.input.shape[1], 1))

        weighted_frame = frame * metric
        self.merged_image += weighted_frame

    def merge_all(self):

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
            self.logger.info("Writing to '{}'.".format(path))
        cv2.imwrite(path, image)
        return True


class DefaultMerger(Merger):
    pass


class CutoffMerger(Merger):

    def _calc_mean(self):
        for frame in self.input.get_frames():
            self.mean_image = frame
            break

    def _calc_metric(self, diff):
        metric = np.sqrt(np.sum(np.square(diff/255), axis=-1))
        metric = cv2.GaussianBlur(metric, (5,5), 1)
        metric = np.piecewise(metric, [metric < 0.1, metric >= 0.1], [0.1/self.input.number_images, 1])
        return metric


class PatchedMeanCutoffMerger(CutoffMerger):
    def _calc_mean(self):
        # for frame in self.input.get_frames():
        #     self.mean_image = frame
        #     break
        width = self.input.shape[1]
        width_left = int(width/2)
        left = self.input.get_frame(-1)[:, :width_left, :]
        right = self.input.get_frame(0)[:, width_left:, :]
        mean = np.concatenate((left, right), axis = 1)
        return mean


class CutOffImages(Merger):

    def _calc_mean(self):
        mean = None
        for frame in self.input.get_frames():
            mean = frame
            break
        return mean

    def _calc_metric(self, diff):
        metric = np.sqrt(np.sum(np.square(diff/255), axis=-1))
        metric = cv2.GaussianBlur(metric, (5, 5), 1)
        metric = np.piecewise(metric, [metric < 0.1, metric >= 0.1], [0, 1])
        return metric

    def merge_frame(self, frame, index):
        diff = self._calc_diff(frame)
        metric = self._calc_metric(diff)
        metric = metric.reshape((self.input.shape[0], self.input.shape[1], 1))
        self.save_image(frame*metric, os.path.join("out", "diff_{:03}.png".format(index)))
