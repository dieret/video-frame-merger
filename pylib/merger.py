#!/usr/bin/env python3

from . import log
import numpy as np
import os.path
import os
import cv2
import time
from . import util
import configobj


class Merger(object):
    """ This class overlays frames and produces an output image. """
    def __init__(self, inpt, config: configobj.ConfigObj):

        self._input = inpt
        self._config = config

        self.name = os.path.splitext(os.path.basename(inpt.path))[0]

        self._logger = log.setup_logger("m")

        self.image_format = self._config["m"]["image_format"]

        # which steps should be saved
        self.save = self._config["m"]["save"]
        # live preview
        self.preview = self._config["m"]["preview"]
        self.preview_max_size = (500, None)  # height, width

        # For convenience:
        self._shape_rgb = inpt.shape  # height x width x 3
        self._shape_scalar = (self._shape_rgb[0], self._shape_rgb[1], 1)
        self._shape = (self._shape_rgb[0], self._shape_rgb[1])

    @property
    def output_dir(self):
        return os.path.join("out", self.name)

    def run(self):
        raise NotImplementedError

    def preview_image(self, image, name="image"):
        new_size = util.new_size(image.shape, self.preview_max_size)

        # Note that cv2.imshow has trouble with floats as image type, so cast
        # it! Also note that image dimensions are width x height, where as the
        # array dimensions are height, width.
        resized = cv2.resize(image, (int(new_size[1]), int(new_size[0]))).astype(np.uint8)

        # Note: waitKey(1) waits 1ms, waitKey(0) actually waits for key
        # Name is the name of the window to be updated/created
        cv2.imshow(name, resized)
        cv2.waitKey(1)

    def save_image(self, image, prefix="", frame_no=None):
        filename = prefix
        if not frame_no is None:
            filename += "_{:04}".format(frame_no)
        if not filename:
            filename = "out"
        filename += ".{}".format(self.image_format)
        path = os.path.join(self.output_dir, filename)

        _dir = os.path.dirname(path)
        if _dir and not os.path.isdir(_dir):
            self._logger.info("Creating dir '{}'.".format(_dir))
            try:
                os.makedirs(_dir)
            except:
                self._logger.error("Could not create director '{}'!".format(_dir))
                raise

        self._logger.info("Writing to '{}'.".format(path))

        cv2.imwrite(path, image)
        return True

    @staticmethod
    def scalar_to_grayscale(scalar):
        normed = scalar / scalar.max() * 255
        return np.concatenate((normed, normed, normed), 2)


# todo: implement metric histograms

class SimpleMerger(Merger):
    """ This merger first calculates a mean (default: average over all frames),
    then, for each frame calculates a difference (default: mean - difference),
    from which we calculate a metric, (e.g. R^3 distance).
    We then sum over all frame * metric and divide by the sum of metric. """

    def __init__(self, inpt, config):
        """ Initialize
        :param inpt: InputData object (from pylib/input)
        """
        super().__init__(inpt, config)

        # preprocess options:

        # todo: put in one config object/config file
        # options: mean, single, patched
        self.mean_strategy = "patched"

        # ** How to convert rgb diff to sclar metric? **
        # options: R3

        # ** Layer (e.g. frame * metric) **
        # Options: normal, overlay
        self.layer_strategy = "normal"

    def calc_mean(self) -> np.ndarray:
        if self.mean_strategy == "mean":
            return sum(self._input.get_frames()) / self._input.number_images

        elif self.mean_strategy == "single":
            # return first frame
            for frame in self._input.get_frames():
                return frame

        elif self.mean_strategy == "patched":
            width = self._input.shape[1]
            width_left = int(width/2)
            left = self._input.get_frame(-1)[:, :width_left, :]
            right = self._input.get_frame(0)[:, width_left:, :]
            return np.concatenate((left, right), axis=1)

        else:
            raise NotImplementedError

    def calc_diff(self, mean, frame):
        return mean - frame

    def calc_metric(self, diff: np.ndarray) -> np.ndarray:
        """ Converts difference to metric
        :param diff: Difference between frame and mean as
            numpy.ndarray(shape=(height, width, 3), np.uint8)
        :return: numpy.ndarray(shape=(height, width, 1), np.float)
            with values between 0 and 1
        """

        conf = self._config["m"]["metric"]

        if conf["strategy"] == "r3":
            metric = np.sqrt(np.sum(np.square(diff/255), axis=-1))
        else:
            raise NotImplementedError

        metric *= conf["intensity"]
        metric += conf["zero"]

        # normalize metric
        metric /= metric.max()

        metric = self.calc_metric_postprocessing(metric)
        shape = (diff.shape[0], diff.shape[1], 1)
        return metric.reshape(shape)

    def calc_metric_postprocessing(self, metric: np.ndarray) -> np.ndarray:
        """ Metric postprocessing (e.g. cutoffs, edge detections etc.)
        :param metric: Metric as numpy.ndarray(shape=(height, width), np.float)
            with values between 0 and 1.
        :return: Metric as numpy.ndarray(shape=(height, width), np.float)
            with values between 0 and 1.
        """

        conf = self._config["m"]["mpp"]

        if "blur" in conf["operations"]:
            metric = cv2.GaussianBlur(metric,
                                      tuple(conf["blur"]["shape"]),
                                      *conf["blur"]["sigmas"])

        if "cutoff" in conf["operations"]:
            # todo: make this take a list of thresholds and values and let
            # us automatically generate this
            metric = np.piecewise(
                metric,
                [
                    metric < conf["cutoff"]["threshold"],
                    metric >= conf["cutoff"]["threshold"]
                ],
                [
                    conf["cutoff"]["min"],
                    conf["cutoff"]["max"]
                ]
            )

        if "edge" in conf["operations"]:
            # todo: options for canny as class variable
            gray = metric * 255
            gray = gray.reshape(self._shape).astype(np.uint8)
            edges = cv2.Canny(gray, 100, 200)
            edges = edges.astype(np.float)
            metric = edges

        # normalize metric
        metric /= metric.max()

        return metric

    def calc_merge(self, sum_layer: np.ndarray, sum_metric: np.ndarray) -> np.ndarray:
        if self.layer_strategy == "overlay":
            merge = sum_layer
        else:
            merge = sum_layer/sum_metric

        # Convert to uint8
        merge[merge < 0.] = 0.
        merge = merge.astype(np.uint8)
        return merge

    def run(self):
        self._logger.debug("Run!")

        mean = self.calc_mean()

        if "mean" in self.save:
            self.save_image(mean, "mean")
        if "mean" in self.preview:
            self.preview_image(mean, "mean")

        sum_metric = np.zeros(shape=self._shape_scalar, dtype=float)
        sum_layer = np.zeros(shape=self._shape_rgb, dtype=float)
        start_time = time.time()
        for index, frame in enumerate(self._input.get_frames()):

            if index >= 1:
                fps = index/(time.time() - start_time)
            else:
                fps=0

            self._logger.debug("Processing frame {:04} (fps: {:02.2f})".format(
                index, fps))

            # ** calculations **

            diff = self.calc_diff(mean, frame)
            metric = self.calc_metric(diff)
            layer = frame * metric
            sum_metric += metric

            if self.layer_strategy == "overlay":
                if index == 0:
                    sum_layer = frame
                else:
                    sum_layer = (1-metric) * sum_layer + layer
            else:
                sum_layer += layer

            if "merge" in self.preview or "merge" in self.save:
                merge = self.calc_merge(sum_metric, sum_layer)

            # ** saving **

            if "frame" in self.save:
                self.save_image(frame, "frame", index)
            if "diff" in self.save:
                self.save_image(diff, "diff", index)
            if "metric" in self.save:
                self.save_image(self.scalar_to_grayscale(metric),
                                "metric", index)
            if "merge" in self.save:
                self.save_image(merge, "merge", index)

            # ** previews **

            if "frame" in self.preview:
                self.preview_image(frame, "frame")
            if "diff" in self.preview:
                self.preview_image(diff, "diff")
            if "metric" in self.preview:
                self.preview_image(self.scalar_to_grayscale(metric), "metric")
            if "merge" in self.preview:
                self.preview_image(merge, "merge")

        if "final" in self.save or "final" in self.preview:
            merge = self.calc_merge(sum_layer, sum_metric)
        if "final" in self.save:
            self.save_image(merge, "final")
        if "final" in self.preview:
            self.preview_image(merge, "final")
