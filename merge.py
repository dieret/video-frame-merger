#!/usr/bin/env python3

import cv2
import os.path
import glob
from util import log


class Merger(object):
    def __init__(self):
        self.summed_images = None
        self.number_images = 0
        self.default_shape = None

        self.logger = log.setup_logger("merger")

    @property
    def mean_image(self):
        return self.summed_images / self.number_images

    def add_image(self, path: str) -> bool:
        if not os.path.exists(path):
            print("Does not exist: '{}'; skip. ".format(path))
            return False

        frame = cv2.imread(path)

        if self.number_images >= 1 and not frame.shape == self.default_shape:
            self.logger.warning(
                "Shapes don't match: '{}' has shape {}, whereas the "
                "first image had '{}'".format(
                    path,
                    frame.shape,
                    self.default_shape))
            return False

        if self.number_images == 0:
            self.summed_images = frame
            self.default_shape = frame.shape
        else:
            self.summed_images += frame

        self.number_images += 1

        return True

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
        m.add_image(path)
    # m.show_image(m.summed_images)
    m.save_image(m.summed_images, "out/sum.png")
    # m.show_image(m.mean_image)
    m.save_image(m.mean_image, "out/mean.png")
    print(m.number_images)

if __name__ == "__main__":
    # m = Merger()
    baxter()
