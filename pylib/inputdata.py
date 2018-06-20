#!/usr/bin/env python3

import shutil
from . import log
from subprocess import call
import cv2
import os.path
import numpy as np


class InputData(object):
    """ This class decides which FrameIterator class to take and also can
    load all images into ram. """
    def __init__(self, path, frame_iterator):
        self.path = path
        self.keep_in_ram = False # I'd be really careful with that one
        self._as_list = None
        self._frame_iterator = frame_iterator(self.path)
        self.logger = log.setup_logger("InputData")

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

    def get_frame(self, index):
        if index < 0:
            # fixme: why the -2
            index = self.number_images - index - 2
        return self._frame_iterator.get_frame(index)

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
        self.logger = log.setup_logger("FrameIterator")

        self.index = 0

    def __iter__(self):
        return self

    def get_frame(self, index=None):
        raise NotImplementedError

    def rewind(self):
        self.logger.debug("Rewinding.")
        self.index = 0
        self._rewind()

    def _rewind(self):
        pass

    def __next__(self):
        frame = self.get_frame()
        if frame is None:
            raise StopIteration

        self.logger.debug("Frame no {}".format(self.index))

        if not frame.shape == self.shape:
            self.logger.warning(
                "Shapes don't match: Frame {} has shape {}, whereas the "
                "first image had '{}'. Skip".format(
                    self.index,
                    frame.shape,
                    self.shape))
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
        return 1.

    def number_images_manual(self):
        num = 0
        self.rewind()
        while True:
            frame = self.get_frame()
            if frame is None:
                break
            else:
                num += 1
        return num


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

    def _rewind(self):
        # doesn't work with gifs:
        # self.opened.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.opened = cv2.VideoCapture(self.path)

    def get_frame(self, index=None):
        if not index:
            okay, frame = self.opened.read()
            if not okay:
                return None
            self.index += 1
        else:
            if 0 <= index < self.number_images:
                self.opened.set(cv2.CAP_PROP_POS_FRAMES, index)
                okay, frame = self.opened.read()
                if not okay:
                    return None
                self.opened.set(cv2.CAP_PROP_POS_FRAMES, self.index)
            else:
                self.logger.error("Out of range!")
                return None

        # note: cv2.imread returns us an array in int8, so we need to
        # convert that.
        # also note that this BGR and not RGB
        frame = frame.astype(np.float)

        return frame

    @property
    def number_images(self):
        if self._number_images is None:
            self._number_images = int(self.opened.get(cv2.CAP_PROP_FRAME_COUNT))
            if not (0 < self._number_images < 1e6):
                # (for gifs)
                self.logger.warning("Automatic retrieval of number of frames "
                                    "failed. Counting manually.")
                self._number_images = self.number_images_manual()
            self.logger.debug("Updated number images to {}.".format(self._number_images))
        return self._number_images

    @property
    def shape(self):
        if self._shape is None:
            self._shape = (
                int(self.opened.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                int(self.opened.get(cv2.CAP_PROP_FRAME_WIDTH)),
                3)
            self.logger.debug("Updated shape to {}".format(self._shape))
        return self._shape

    @property
    def fps(self):
        if self._fps is None:
            self._fps = self.opened.get(cv2.CAP_PROP_FPS)
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

    def _get_frame(self, index):
        return cv2.imread(self.video_files[index]).astype(np.float)

    def get_frame(self, index=None):
        if not index:
            if not self.index < len(self.video_files):
                return None
            frame = self._get_frame(self.index)
            self.index += 1
        else:
            return self._get_frame(index)
        return frame

    @property
    def number_images(self):
        return len(self.video_files)

    @property
    def shape(self):
        if self._shape is None:
            self._shape = self._get_frame(0).shape
        return self._shape


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
