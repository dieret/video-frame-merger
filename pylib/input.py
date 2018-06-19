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
        self._as_list = None
        self._frame_iterator = frame_iterator(self.path)
        self.logger = log.setup_logger("Input")

    def get_frames(self):
        if self.keep_in_ram and self._as_list:
            return self._as_list

        self.logger.debug("Getting frames.")

        self._frame_iterator.rewind()

        if self.keep_in_ram:
            self.logger.debug("Reading into ram.")
            self._as_list = list(self._frame_iterator)
            return self._as_list
        else:
            return self._frame_iterator

    @property
    def number_images(self):
        return self._frame_iterator.number_images

    @property
    def shape(self):
        return self._frame_iterator.shape

    @property
    def fps(self):
        return self._frame_iterator.fps


class FrameIterator(object):
    """ This class will be subclassed by implementing the get_frame method.
    Iterate over this class to get all frames of vide/input. """
    def __init__(self):
        self.default_shape = None
        self.logger = log.setup_logger("FrameIterator")

        self.index = 0

    def __iter__(self):
        return self

    def get_frame(self):
        raise NotImplementedError

    def rewind(self):
        raise NotImplementedError

    def __next__(self):
        frame = self.get_frame()
        if frame is None:
            raise StopIteration

        self.logger.debug("Frame no {}".format(self.index))

        if frame.shape is None:
            self.default_shape = frame.shape

        if not frame.shape == self.shape:
            self.logger.warning(
                "Shapes don't match: Frame {} has shape {}, whereas the "
                "first image had '{}'. Skip".format(
                    self.index,
                    frame.shape,
                    self.default_shape))
            return self.__next__

        return frame

    @property
    def number_images(self):
        raise NotImplementedError

    @property
    def shape(self):
        raise NotImplementedError

    @property
    def fps(self):
        raise NotImplementedError


class VideoFrameIterator(FrameIterator):
    """ This class gets all the frames of a video on the fly. """
    def __init__(self, path: str):
        super().__init__()
        if not os.path.exists(path):
            self.logger.critical("File does not exist: '{}'".format(self.path))
            raise ValueError
        self.path = path
        self.opened = cv2.VideoCapture(self.path)

        self._number_images = None
        self._shape = None
        self._fps = None

    def rewind(self):
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

    @property
    def number_images(self):
        if self._number_images is None:
            self._number_images = int(self.opened.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
        return self._number_images

    @property
    def shape(self):
        if self._shape is None:
            self._shape = (
                int(self.opened.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)),
                int(self.opened.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)),
                3)
        return self._shape

    @property
    def fps(self):
        if self._fps is None:
            self._fps = self.opened.get(cv2.cv.CV_CAP_PROP_FPS)
        return self._fps


class SingleFramesIterator(FrameIterator):
    """ This class takes a list of files as frames. """
    def __init__(self, video_files):
        super().__init__()
        self.burst_base_dir = os.path.join("data", "burst")
        self.index = 0
        self.video_files = video_files
        if not self.video_files:
            self.logger.critical("No input files.")
            raise ValueError()

        self._shape = None

    def rewind(self):
        self.index = 0

    def _get_frame(self, index):
        return cv2.imread(self.video_files[index]).astype(np.float)

    def get_frame(self):
        if not self.index < len(self.video_files):
            return None
        frame = self._get_frame(self.index)
        self.index += 1
        return frame

    @property
    def number_images(self):
        return len(self.video_files)

    @property
    def shape(self):
        if self._shape is None:
            self._shape =  self._get_frame(0).shape
        return self._shape

    @property
    def fps(self):
        # we really don't know
        return 1.


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

        self._burst_file()

        self.video_files = self._listdir_paths(self.burst_subfolder)

        super().__init__(self.video_files)

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
        os.makedirs(folder)

    def _listdir_paths(self, folder):
        if not os.path.isdir(folder):
            self.logger.critical("'{}' is not a folder. Abort!".format(folder))
            raise ValueError
        return [os.path.join(folder, filename) for filename in os.listdir(folder)]