#!/usr/bin/env python3

from . import log
import numpy as np
import os.path
import os
import cv2
import collections


class Merger(object):
    """ This class overlays frames and produces an output image. """
    def __init__(self, inpt):

        self.input = inpt
        self.logger = log.setup_logger("Merger")

        # For convenience:
        self.shape_rgb = inpt.shape # height x width x 3
        self.shape_scalar = (self.shape_rgb[0], self.shape_rgb[1], 1)

    def run(self):
        raise NotImplementedError

    @staticmethod
    def show_image(image):
        cv2.imshow('Image', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # more configuration
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

    @staticmethod
    def scalar_to_grayscale(scalar):
        normed = scalar / scalar.max() * 255
        return np.concatenate((normed, normed, normed), 2)


class SimpleMerger(Merger):
    def __init__(self, inpt, fifo_length=1):
        super().__init__(inpt)

        # Progress images to save
        # mean, diff, metric, merge, final
        self.save = ["final"]

        # these values will be continuously updated in the self.run loop
        # This will allow us more flexibility than using parameters for the
        # class functions.
        self.index = 0
        self.frame = None
        self.mean = None
        self.diff = None
        self.metric = None
        self.final = None

        self.sum_metric = np.zeros(shape=self.shape_scalar, dtype=float)
        self.sum_layers = np.zeros(shape=self.shape_rgb, dtype=float)

    def calc_mean(self):
        self.mean = sum(self.input.get_frames()) / self.input.number_images

    def calc_diff(self):
        self.diff =  self.mean - self.frame

    def calc_metric(self):
        # change intensity to increase emphasis of outliers
        intensity = 500
        metric = 1 + intensity * np.sqrt(np.sum(np.square(self.diff/255), axis=-1))
        metric = cv2.GaussianBlur(metric, (5, 5), 0)
        self.metric = metric.reshape(self.shape_scalar)

    def calc_sum_metric(self):
        self.sum_metric += self.metric

    def calc_sum_layers(self):
        self.sum_layers += self.frame * self.metric

    def calc_final(self):
        self.final = self.sum_layers / self.sum_metric

    def run(self):

        self.logger.debug("Run!")

        mean = self.calc_mean()
        if "mean" in self.save:
            self.save_image(mean, "out/mean.png")

        for self.index, self.frame in enumerate(self.input.get_frames()):
            self.calc_diff()
            self.calc_metric()
            self.calc_sum_metric()
            self.calc_sum_layers()

            if "diff" in self.save:
                self.save_image(
                    self.diff,
                    os.path.join("out", "diff_{:03}.png".format(self.index)))
            if "metric" in self.save:
                self.save_image(
                    self.scalar_to_grayscale(self.metric),
                    os.path.join("out", "metric_{:03}.png".format(self.index)))
            if "merge" in self.save:
                self.calc_final()
                self.save_image(
                    self.final,
                    os.path.join("out", "merge_{:03}.png".format(self.index)))

        self.calc_final()
        if "final" in self.save:
            self.save_image(
                self.final,
                os.path.join("out", "out.png"))


class SimpleMeanMerger(SimpleMerger):

    def __init__(self, inpt):
        super().__init__(inpt)

    def calc_mean(self):
        for frame in self.input.get_frames():
            self.mean = frame
            return

# class FifoMerger(object):
#     """ This class overlays frames and produces an output image. """
#     def __init__(self, inpt, fifo_length=1):
#
#         self.input = inpt
#
#         # height x width x 3
#         self.shape_rgb = inpt.shape
#         # height x width x 1
#         self.shape_scalar = (self.shape_rgb[0], self.shape_rgb[1], 1)
#
#         self.logger = log.setup_logger("merger")
#
#         # currently processed frame
#         self.frame_no = 0
#
#         # FIFOs = First in first out: Keep copies of the last n frames
#         self.fifo_frame = collections.deque(maxlen=fifo_length)
#         self.fifo_mean = collections.deque(maxlen=fifo_length)
#         self.fifo_diff = collections.deque(maxlen=fifo_length)
#         self.fifo_metric = collections.deque(maxlen=fifo_length)
#         self.fifo_merged = collections.deque(maxlen=fifo_length)
#
#
#         # Sums
#         self.sum_metric = np.zeros(self.shape_scalar)
#         self.sum_layers = np.zeros(self.shape_scalar)
#
#         # Config
#         self.save_diff = False
#         self.save_mean = True
#         self.save_metric = False



class CutoffMerger(SimpleMerger):

    def __init__(self, inpt):
        super().__init__(inpt)

    def calc_metric(self):
        metric = np.sqrt(np.sum(np.square(self.diff/255), axis=-1))
        metric = cv2.GaussianBlur(metric, (5, 5), 1)
        metric = np.piecewise(metric, [metric < 0.1, metric >= 0.1], [0.1/self.input.number_images, 1])
        self.metric = metric.reshape(self.shape_scalar)


class SimpleMeanCutoffMerger(CutoffMerger):

    def __init__(self, inpt):
        super().__init__(inpt)

    def calc_mean(self):
        for frame in self.input.get_frames():
             self.mean_image = frame
             return


class PatchedMeanCutoffMerger(SimpleMerger):

    def __init__(self, inpt):
        super().__init__(inpt)

    def calc_mean(self):
        width = self.input.shape[1]
        width_left = int(width/2)
        left = self.input.get_frame(-1)[:, :width_left, :]
        right = self.input.get_frame(0)[:, width_left:, :]
        self.mean = np.concatenate((left, right), axis = 1)


class OverlayMerger(PatchedMeanCutoffMerger):

    def __init__(self, inpt):
        super().__init__(inpt)
        self.save.append("merge")

    def calc_metric(self):
        # difference to PatchedMeanCutoffMerger: Take 0, not just small value
        metric = np.sqrt(np.sum(np.square(self.diff/255), axis=-1))
        metric = cv2.GaussianBlur(metric, (5, 5), 1)
        metric = np.piecewise(metric, [metric < 0.1, metric >= 0.1], [0, 1])
        self.metric = metric.reshape(self.shape_scalar)

    def calc_sum_layers(self):
        if self.index == 0:
            self.sum_layers = self.frame
        else:
            self.sum_layers = (1-self.metric) * self.sum_layers + self.metric * self.frame

    def calc_final(self):
        self.final = self.sum_layers
