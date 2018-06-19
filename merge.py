#!/usr/bin/env python3

import cv2
import os
import glob
import shutil
from util import log
import numpy as np
from subprocess import call


class FrameIterator(object):
    def __init__(self):
        self.default_shape = None
        self.number_images = 0
        self.logger = log.setup_logger("FrameIterator")

    def __iter__(self):
        return self

    def get_frame(self):
        raise NotImplementedError

    def __next__(self):
        frame = self.get_frame()
        if frame is None:
            raise StopIteration

        self.logger.debug("Frame no {}".format(self.number_images))

        if self.number_images == 0:
            self.default_shape = frame.shape

        if self.number_images >= 1 and not frame.shape == self.default_shape:
            self.logger.warning(
                "Shapes don't match: Frame {} has shape {}, whereas the "
                "first image had '{}'. Skip".format(
                    self.number_images,
                    frame.shape,
                    self.default_shape))
            return self.__next__

        self.number_images += 1

        return frame


class VideoFrameIterator(FrameIterator):
    def __init__(self, path: str):
        super().__init__()
        if not os.path.exists(path):
            self.logger.critical("File does not exist: '{}'".format(self.path))
            raise ValueError
        self.path = path
        self.opened = cv2.VideoCapture(self.path)

    def get_frame(self):
        okay, frame = self.opened.read()
        if not okay:
            return None

        # note: cv2.imread returns us an array in int8, so we need to
        # convert that.
        # also note that this BGR and not RGB
        frame = frame.astype(np.float)

        return frame


class SingleFramesIterator(FrameIterator):
    def __init__(self, video_files):
        super().__init__()
        self.burst_base_dir = os.path.join("data", "burst")
        self.index = 0
        self.number_images = 0
        self.video_files = video_files
        if not self.video_files:
            self.logger.critical("No input files.")
            raise ValueError()

    def get_frame(self):
        if not self.index < len(self.video_files):
            return None

        frame = cv2.imread(self.video_files[self.index]).astype(np.float)

        self.index += 1

        return frame


class BurstFrameIterator(SingleFramesIterator):
    def __init__(self, path):
        self.logger = log.setup_logger("FrameIterator")
        self.path = path
        self.burst_base_dir = os.path.join("data", "burst")
        self.burst_subfolder = os.path.join(
            self.burst_base_dir,
            os.path.splitext(os.path.basename(self.path))[0])
        self.video_files = self._listdir_paths(self.burst_subfolder)

        self._burst_file()

        super().__init__(self.video_files)

        if not self.video_files:
            self.logger.critical("No input files.")
            raise ValueError()

    def _burst_file(self):
        # todo: maybe implement check if we even need that
        self.logger.debug("Bursting '{}' to '{}'.".format(self.path, self.burst_subfolder))
        self._get_clean_folder(self.burst_subfolder)
        destination = os.path.join(self.burst_subfolder,
                                   os.path.splitext(os.path.basename(self.path))[0] + "%5d.png")
        call(["ffmpeg", "-loglevel", "warning", "-i", self.path, destination])

    @staticmethod
    def _get_clean_folder(folder):
        """ Clear folder if existent, create if nonexistent """
        if os.path.exists(folder):
            shutil.rmtree(folder)
        os.mkdir(folder)

    @staticmethod
    def _listdir_paths(folder):
        return [os.path.join(folder, filename) for filename in os.listdir(folder)]


class Input(object):
    def __init__(self, path, frame_iterator=VideoFrameIterator):
        self.path = path
        self.keep_in_ram = False
        self.as_list = None
        self.frame_iterator = frame_iterator
        self.logger = log.setup_logger("Input")

    def get_frames(self):
        if self.keep_in_ram and self.as_list:
            return self.as_list

        self.logger.debug("Getting frames.")

        iter = self.frame_iterator(self.path)

        if self.keep_in_ram:
            self.logger.debug("Reading into ram.")
            self.as_list = list(iter)
            return self.as_list
        else:
            return iter


class Merger(object):
    def __init__(self, input):

        self.input = input

        self.logger = log.setup_logger("merger")

        self.number_images = 0
        self.merged_images = None
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
        self.merged_images = np.ndarray(shape=self.default_shape, dtype=np.float)

        for frame in self.input.get_frames():
            diff = (self.mean_image - frame)/255

            # change intensity to increase emphasis of outliers
            intensity = 10
            metric = 1 + intensity*np.sqrt(np.sum(np.square(diff), axis=-1))

            self.sum_weights += metric

            # blurring in space as an attemptas an attemptas an attempt to reduce noiseee
            # metric = cv2.GaussianBlur(metric, (5,5), 0)

            # metric is not a width x size array. In order to multiply it to
            # the width x size x 3 array of the picture, we must broad cast it
            # to width x size x 3, so numpy gets this :(
            metric = metric.reshape(tuple(list(self.default_shape)[:-1]+[1]))

            weighted_frame = frame * metric

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

        print("writing " + path)
        cv2.imwrite(path, image)

    def get_merged_image(self):
        return self.merged_images / self.sum_weights.reshape(tuple(list(self.default_shape)[:-1]+[1]))


if __name__ == "__main__":
    # m = Merger()
    # burst_and_merge_gifs()
    inpt = Input("data/giflibrary/baxter.gif", BurstFrameIterator)
    m = Merger(inpt)
    m.calc_merged()
