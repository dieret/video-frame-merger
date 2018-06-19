#!/usr/bin/env python3

import cv2
import os
import glob
import shutil
from util import log
import numpy as np
from subprocess import call


class FrameIterator(object):
    def __init__(self, path: str):
        self.path = path
        self.default_shape = None
        self.number_images = 0
        self.logger = log.setup_logger("FrameIterator")

    def __iter__(self):
        return self

    def __next__(self):
        raise NotImplementedError


class OntheflyFrameIterator(FrameIterator):
    def __init__(self, path: str):
        super().__init__(path)
        if not os.path.exists(self.path):
            self.logger.critical("File does not exist: '{}'".format(self.path))
            raise ValueError
        self.opened = cv2.VideoCapture(self.path)

    def __next__(self):
        okay, frame = self.opened.read()
        if not okay:
            raise StopIteration

        if self.number_images == 0:
            self.default_shape = frame.shape

        # self.logger.debug("Frame no {}".format(self.number_images))

        self.number_images += 1

        # note: cv2.imread returns us an array in int8, so we need to
        # convert that.
        # also note that this BGR and not RGB
        frame = frame.astype(np.float)

        if self.number_images >= 1 and not frame.shape == self.default_shape:
            self.logger.warning(
                "Shapes don't match: Frame {} has shape {}, whereas the "
                "first image had '{}'. Skip".format(
                    self.number_images,
                    frame.shape,
                    self.default_shape))
            return self.__next__

        return frame


class Input(object):
    def __init__(self, path, frame_iterator=OntheflyFrameIterator):
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

# def burst_destination_folder():
#     return os.path.join("data", "burst")
#
# def video_source_folder():
#     return os.path.join("data", "giflibrary")
#
# def burst_gifs():
#     get_clean_folder(burst_destination_folder())
#     files = get_video_files(video_source_folder())
#     for file in files:
#         burst_file(file)
#
# def file_extension(file):
#     return file.split(".")[-1]
#
# def strip_extension(file):
#     return ".".join(file.split(".")[0:-1])
#
# def get_video_files(folder):
#     files = os.listdir(folder)
#     return [file for file in files if file_extension(file) in {"gif"}]
#
# # clear folder if existent, create if nonexistent
# def get_clean_folder(folder):
#     if os.path.exists(folder):
#         shutil.rmtree(folder)
#     os.mkdir(folder)
#
# # burst file into a clean subfolder of burst_folder
# def burst_file(file):
#     subfolder = os.path.join(burst_destination_folder(), strip_extension(file))
#     get_clean_folder(subfolder)
#     destination = os.path.join(subfolder, strip_extension(file) + "%5d.png")
#     call(["ffmpeg", "-i", os.path.join(video_source_folder(), file), destination])
#
#
# def burst_and_merge_gifs():
#     burst_gifs()
#     for file in get_video_files(video_source_folder()):
#         merge_gif(file)
#
# # copy of baxter()
# def merge_gif(file):
#     image_name = strip_extension(file)
#     burst_folder = os.path.join(burst_destination_folder(), image_name)
#
#     # note: unsorted, but we don't care about that right now.
#     paths = glob.glob(os.path.join(burst_folder, "*.png"))
#     m = Merger()
#     for path in paths:
#         m.add_to_mean(path)
#
#     for path in paths:
#         m.add_to_merged(path)
#     m.save_image(m.get_merged_image(), "out/" + image_name + ".png")
#
#
# def baxter():
#     base_path = os.path.join("data", "burst")
#     # note: unsorted, but we don't care about that right now.
#     paths = glob.glob(os.path.join(base_path, "baxter-*.png"))
#     m = Merger()
#     for path in paths:
#         m.add_to_mean(path)
#     m.save_image(m.mean_image, "out/mean.png")
#     for path in paths:
#         m.add_to_merged(path)
#     m.save_image(m.merged_images, "out/merged.png")
#     m.save_image(m.get_merged_image(), "out/merged-normed.png")

if __name__ == "__main__":
    # m = Merger()
    # burst_and_merge_gifs()
    inpt = Input("data/giflibrary/baxter.gif")
    m = Merger(inpt)
    m.calc_merged()
