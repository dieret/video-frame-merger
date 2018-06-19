#!/usr/bin/env python3

import cv2
import os
import glob
import shutil
from util import log
import numpy as np
from subprocess import call

class Merger(object):
    def __init__(self):
        self.summed_images = None
        self.number_images = 0
        self.merged_images = None
        self.default_shape = None
        self.sum_weights = None

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
            self.sum_weights = np.ndarray(shape=tuple(list(self.default_shape)[:-1]),
                                          dtype=np.float)

        diff = (self.mean_image - frame)/255
        metric = 1 + 600*np.sqrt(np.sum(np.square(diff), axis=-1))
        # max 3
        self.sum_weights += metric

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

        cv2.imwrite(path, image)

    def get_merged_image(self):
        return self.merged_images / self.sum_weights.reshape(tuple(list(self.default_shape)[:-1]+[1]))

def burst_destination_folder():
    return os.path.join("data", "burst")

def video_source_folder():
    return os.path.join("data", "giflibrary")

def burst_gifs():
    get_clean_folder(burst_destination_folder())
    files = get_video_files(video_source_folder())
    for file in files:
        burst_file(file)

def file_extension(file):
    return file.split(".")[-1]

def strip_extension(file):
    return ".".join(file.split(".")[0:-1])

def get_video_files(folder):
    files = os.listdir(folder)
    return [file for file in files if file_extension(file) in {"gif"}]

# clear folder if existent, create if nonexistent    
def get_clean_folder(folder):
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.mkdir(folder)

# burst file into a clean subfolder of burst_folder
def burst_file(file):
    subfolder = os.path.join(burst_destination_folder(), strip_extension(file))
    get_clean_folder(subfolder)
    destination = os.path.join(subfolder, file + "%5d.png")
    call(["ffmpeg", "-i", os.path.join(video_source_folder(), file), destination]) 


def burst_and_merge_gifs():
    burst_gifs()
    for file in get_video_files(video_source_folder()):
        merge_gif(file)

# copy of baxter()
def merge_gif(file):
    image_name = strip_extension(file)
    burst_folder = os.path.join(burst_destination_folder(), image_name)
    
    # note: unsorted, but we don't care about that right now.
    paths = glob.glob(os.path.join(burst_folder, "*.png"))
    m = Merger()
    for path in paths:
        m.add_to_mean(path)
    m.save_image(m.mean_image, "out/mean.png")
    for path in paths:
        m.add_to_merged(path)
    m.save_image(m.merged_images, "out/" + image_name + "_merged.png")
    m.save_image(m.get_merged_image(), "out/" + image_name + "_merged-normed.png")
    

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
    m.save_image(m.get_merged_image(), "out/merged-normed.png")

if __name__ == "__main__":
    # m = Merger()
    burst_and_merge_gifs()
