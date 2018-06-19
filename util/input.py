#!/usr/bin/env python3

import shutil
from . import log
from subprocess import call
import cv2
import os.path
import numpy as np


class Input(object):
    """ This class decides which FrameIterator class to take and also can
    load all images into ram. """
    def __init__(self, path, frame_iterator):
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


class FrameIterator(object):
    """ This class will be subclassed by implementing the get_frame method.
    Iterate over this class to get all frames of vide/input. """
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
    """ This class gets all the frames of a video on the fly. """
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
    """ This class takes a list of files as frames. """
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
    """ This class first bursts a video file into single images, then
    uses SingleFramesIterator to iterate over them. """
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


