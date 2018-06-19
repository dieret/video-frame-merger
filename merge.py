#!/usr/bin/env python3

import os
from util.input import *
import numpy as np
import cv2
import argparse
import util.input


class Merger(object):
    """ This class overlays frames and produces an output image. """
    def __init__(self, input):

        self.input = input

        self.logger = log.setup_logger("merger")

        self.number_images = 0
        self.merged_image = None
        self.sum_weights = None
        self.mean_image = None
        self.default_shape = None

    def calc_mean(self):
        self.logger.debug("Calculating mean.")
        self.number_images = 0
        for frame in self.input.get_frames():
            if frame is None:
                self.logger.warning("Skipping frame.")
                continue
            if self.mean_image is None:
                self.mean_image = frame
                self.default_shape = frame.shape
            else:
                self.mean_image += frame
            self.number_images += 1

        self.mean_image /= self.number_images

    def calc_merged(self):

        if self.mean_image is None:
            self.calc_mean()

        self.logger.debug("Calculating merged.")


        self.sum_weights = np.ndarray(shape=tuple(list(self.default_shape)[:-1]),
                                          dtype=np.float)
        self.merged_image = np.ndarray(shape=self.default_shape, dtype=np.float)

        for frame in self.input.get_frames():
            diff = (self.mean_image - frame)/255

            # change intensity to increase emphasis of outliers
            intensity = 10
            metric = 1 + intensity*np.sqrt(np.sum(np.square(diff), axis=-1))

            # blurring in space as an attemptas an attemptas an attempt to reduce noiseee
            metric = cv2.GaussianBlur(metric, (5,5), 0)

            self.sum_weights += metric

            # metric is not a width x size array. In order to multiply it to
            # the width x size x 3 array of the picture, we must broad cast it
            # to width x size x 3, so numpy gets this :(
            metric = metric.reshape(tuple(list(self.default_shape)[:-1]+[1]))

            weighted_frame = frame * metric

            self.merged_image += weighted_frame

        self.merged_image = self.merged_image / self.sum_weights.reshape(tuple(list(self.default_shape)[:-1] + [1]))


    @staticmethod
    def show_image(image):
        cv2.imshow('Image', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    @staticmethod
    def save_image(image, path=os.path.join("out", "out.png")):
        dir = os.path.dirname(path)
        if dir and not os.path.exists(dir):
            os.mkdir(dir)

        print("writing " + path)
        cv2.imwrite(path, image)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        dest="input_path",
        type=str,
        nargs="+",
        help="Input video file")
    parser.add_argument(
        "-i",
        "--iterator",
        default="VideoFrameIterator",
        help="Frame iterator.. Currently 3 options: "
             "VideoFrameIterator, SinglFramesIterator, "
             "BurstFrameIterator.")
    parser.add_argument(
        "-s",
        "--show",
        action="store_true",
        default=False,
        help="Show picture upon completion."
    )

    args = parser.parse_args()

    if args.iterator != "SingleFramesIterator":
        assert(len(args.input_path)) == 1
        args.input_path = args.input_path[0]

    inpt = Input(args.input_path, getattr(util.input, args.iterator))
    m = Merger(inpt)
    m.calc_merged()

    if args.show:
        m.show_image(m.merged_image)

    m.save_image(m.merged_image)
