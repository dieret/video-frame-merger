#!/usr/bin/env python3

import cv2
import os.path
import glob
from util import log
import numpy as np


class Merger(object):
    def __init__(self):
        self.summed_images = None
        self.number_images = 0
        self.merged_images = None
        self.default_shape = None

        self.logger = log.setup_logger("merger")


    # todo: probably want some other way to do this, because we recalculate it every time like this
    @property
    def mean_image(self):
        return self.summed_images / self.number_images

    def get_image(self, path: str):
        if not os.path.exists(path):
            self.logger.warning("Does not exist: '{}'; skip. ".format(path))
            return False

        # note: cv2.imread returns us an array in int8, so we need to
        # convert that.
        # also note that this BGR and not RGB
        frame = cv2.imread(path).astype(np.int32)

        if self.number_images >= 1 and not frame.shape == self.default_shape:
            self.logger.warning(
                "Shapes don't match: '{}' has shape {}, whereas the "
                "first image had '{}'".format(
                    path,
                    frame.shape,
                    self.default_shape))
            return None

        return frame

    def add_to_mean(self, path: str) -> bool:

        frame = self.get_image(path)
        if frame is None:
            return False

        if self.number_images == 0:
            self.summed_images = frame
            self.default_shape = frame.shape
        else:
            self.summed_images += frame

        self.number_images += 1

        return True

    def add_to_merged(self, path: str):

        frame = self.get_image(path)
        if frame is None:
            return False

        if self.merged_images is None:
            self.merged_images = np.ndarray(shape=self.default_shape, dtype=np.float)

        diff = self.mean_image - frame
        metric = np.sqrt(np.sum(np.square(diff), axis=-1))
        metric /= 255

        # b = frame[:,:,0].astype(np.float)
        # g = frame[:,:,1].astype(np.float)
        # r = frame[:,:,2].astype(np.float)
        #
        # b *= metric
        # g *= metric
        # r *= metric
        #
        # print(b.shape, g.shape, r.shape)
        #
        # weighted_frame = np.ndarray((b, g, r))
        #
        # print(weighted_frame.shape)

        self.merged_images += weighted_frame


    @staticmethod
    def show_image(image):
        cv2.imshow('Image', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    @staticmethod
    def save_image(image, path ="out/out.png"):
        dir = os.path.dirname(path)
        if dir and not os.path.exists(dir):
            os.mkdir(dir)

        cv2.imwrite(path, image)


def baxter():
    base_path = os.path.join("data", "burst")
    # note: unsorted, but we don't care about that right now.
    paths = glob.glob(os.path.join(base_path, "baxter-*.png"))
    m = Merger()
    for path in paths:
        m.add_to_mean(path)
    m.save_image(m.mean_image, "out/mean.png")
    for path in paths:
        m.add_to_merged(path)
    m.save_image(m.merged_images, "out/merged.png")

if __name__ == "__main__":
    # m = Merger()
    baxter()
