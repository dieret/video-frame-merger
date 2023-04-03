#!/usr/bin/env python3

from . import log
import numpy as np
import os.path
import os
import cv2
import time
from . import util
import configobj

# todo: use for loops for all of the operations, so that we can check if the
# operation is supported and also can apply the same operations in different
# orders and more than once

# todo: add automatic video output, so that we don't have to use ffmpeg


class Merger(object):
    """This class overlays frames and produces an output image."""

    def __init__(self, inpt, config: configobj.ConfigObj):
        self._input = inpt
        self._config = config

        self.name = os.path.splitext(os.path.basename(inpt.path))[0]

        self._logger = log.setup_logger("m")

        # which steps should be saved
        self.save = self._config["m"]["save"]
        # live preview
        self.preview = self._config["m"]["preview"]

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
        # convert to grayscale if needed
        if len(image.shape) == 2 or (
            len(image.shape) == 3 and image.shape[2] == 1
        ):
            image = self.scalar_to_grayscale(image)

        new_size = util.new_size(
            image.shape, self._config["m"]["preview_max_size"]
        )

        # Note that cv2.imshow has trouble with floats as image type, so cast
        # it! Also note that image dimensions are width x height, where as the
        # array dimensions are height, width.
        resized = cv2.resize(image, (int(new_size[1]), int(new_size[0]))).astype(
            np.uint8
        )

        # Note: waitKey(1) waits 1ms, waitKey(0) actually waits for key
        # Name is the name of the window to be updated/created
        cv2.imshow(name, resized)
        cv2.waitKey(1)

    def save_image(self, image, prefix="", frame_no=None):
        filename = prefix
        if frame_no is not None:
            filename += "_{:04}".format(frame_no)
        if not filename:
            filename = "out"
        filename += ".{}".format(self._config["m"]["image_format"])
        path = os.path.join(self.output_dir, filename)

        _dir = os.path.dirname(path)
        if _dir and not os.path.isdir(_dir):
            self._logger.info("Creating dir '{}'.".format(_dir))
            try:
                os.makedirs(_dir)
            except Exception as e:
                self._logger.error(f"Could not create director '{_dir}': {e}!")
                raise

        self._logger.info("Writing to '{}'.".format(path))

        if len(image.shape) == 2 or (
            len(image.shape) == 3 and image.shape[2] == 1
        ):
            image = self.scalar_to_grayscale(image)
        cv2.imwrite(path, image)
        return True

    @staticmethod
    def scalar_to_grayscale(scalar):
        assert 2 <= len(scalar.shape) <= 3
        if scalar.shape == 2:
            scalar = scalar.reshape((*scalar.shape, 1))
        assert len(scalar.shape) == 3 and scalar.shape[2] == 1
        normed = scalar / scalar.max() * 255
        return np.concatenate((normed, normed, normed), 2)


# todo: implement metric histograms


class SimpleMerger(Merger):
    """This merger first calculates a mean (default: average over all frames),
    then, for each frame calculates a difference (default: mean - difference),
    from which we calculate a metric, (e.g. R^3 distance).
    We then sum over all frame * metric and divide by the sum of metric."""

    def __init__(self, inpt, config):
        """Initialize
        :param inpt: InputData object (from pylib/input)
        """
        super().__init__(inpt, config)

    def calc_mean(self) -> np.ndarray:
        conf = self._config["m"]["mean"]
        if conf["strategy"] == "mean":
            return sum(self._input.get_frames()) / self._input.number_images

        elif conf["strategy"] == "single":
            # return first frame
            return self._input.get_frame(conf["single"]["no"])

        elif conf["strategy"] == "patched":
            width = self._input.shape[1]
            width_left = int(width * conf["patched"]["fraction"])
            left = self._input.get_frame(conf["patched"]["no1"])[
                :, :width_left, :
            ]
            right = self._input.get_frame(conf["patched"]["no2"])[
                :, width_left:, :
            ]
            return np.concatenate((left, right), axis=1)

        else:
            raise NotImplementedError

    def calc_diff(self, mean, frame):
        conf = self._config["m"]["diff"]

        diff = mean - frame

        if "median" in conf["operations"]:
            # todo: avoid costly transformation
            # need uin8 here
            diff = diff.astype(np.uint8)
            diff = cv2.medianBlur(diff, conf["median"]["size"])

        return diff

    def calc_metric(self, diff: np.ndarray) -> np.ndarray:
        """Converts difference to metric
        :param diff: Difference between frame and mean as
            numpy.ndarray(shape=(height, width, 3), np.uint8)
        :return: numpy.ndarray(shape=(height, width, 1), np.float)
            with values between 0 and 1
        """

        conf = self._config["m"]["metric"]

        if conf["strategy"] == "euclidean":
            # no need for /255 since we norm afterwards
            metric = np.sqrt(np.sum(np.square(diff), axis=-1))
        elif conf["strategy"] == "manhattan":
            metric = np.sum(np.abs(diff), axis=-1)
        elif conf["strategy"] == "max":
            metric = np.abs(diff).max(axis=-1)
        elif conf["strategy"] == "p":
            # fixme: not working yet
            metric = np.power(
                np.sum(np.power(diff, conf["p"]["p"]), axis=-1),
                1.0 / conf["p"]["p"],
            )
        else:
            raise NotImplementedError

        metric *= conf["intensity"]
        metric += conf["zero"]

        # normalize metric
        mm = metric.max()
        if mm != 0:
            metric /= mm

        metric = self.calc_metric_postprocessing(metric)
        shape = (diff.shape[0], diff.shape[1], 1)
        return metric.reshape(shape)

    def calc_metric_postprocessing(self, metric: np.ndarray) -> np.ndarray:
        """Metric postprocessing (e.g. cutoffs, edge detections etc.)
        :param metric: Metric as numpy.ndarray(shape=(height, width), np.float)
            with values between 0 and 1.
        :return: Metric as numpy.ndarray(shape=(height, width), np.float)
            with values between 0 and 1.
        """

        conf = self._config["m"]["mpp"]

        if "gauss" in conf["operations"]:
            metric = cv2.GaussianBlur(
                metric, tuple(conf["gauss"]["shape"]), *conf["gauss"]["sigmas"]
            )

        if "open" in conf["operations"]:
            kernel = np.ones(conf["open"]["kernel"], np.uint8)
            metric = cv2.morphologyEx(metric, cv2.MORPH_OPEN, kernel)

        if "cutoff" in conf["operations"]:
            # todo: make this take a list of thresholds and values and let
            # us automatically generate this
            metric = np.piecewise(
                metric,
                [
                    metric < conf["cutoff"]["threshold"],
                    metric >= conf["cutoff"]["threshold"],
                ],
                [conf["cutoff"]["min"], conf["cutoff"]["max"]],
            )

        if "edge" in conf["operations"]:
            gray = metric * 255
            gray = gray.reshape(self._shape).astype(np.uint8)
            edges = cv2.Canny(
                gray, conf["edge"]["canny1"], conf["edge"]["canny2"]
            )
            edges = edges.astype(np.float)
            metric = edges

        if "dilate" in conf["operations"]:
            kernel = np.ones(conf["dilate"]["kernel"], np.uint8)
            metric = cv2.dilate(metric, kernel)

        # normalize metric
        mm = metric.max()
        if mm != 0:
            metric /= mm

        return metric

    def calc_merge(
        self, sum_layer: np.ndarray, sum_metric: np.ndarray
    ) -> np.ndarray:
        conf = self._config["m"]["overlay"]

        if conf["strategy"] in ["overlay", "overlaymean"]:
            merge = sum_layer
        elif conf["strategy"] == "add":
            # todo: Taking care of zeros to avoid zero division.
            # Wherever the metric is, we set the layer to zero as well.
            merge = sum_layer / sum_metric
        else:
            raise ValueError("Unknown parameter.")

        # todo: add normalization option

        # Convert to uint8
        merge[merge <= 0] = 0
        merge = merge.astype(np.uint8)
        return merge

    def calc_layer(self, frame, metric):
        conf = self._config["m"]["layer"]
        layer = conf["multiply"] * frame * metric
        layer += conf["add"]

        if "median" in conf["operations"]:
            layer = layer.astype(np.uint8)
            layer = cv2.medianBlur(layer, conf["median"]["size"])

        if "gauss" in conf["operations"]:
            layer = cv2.GaussianBlur(
                layer, tuple(conf["gauss"]["shape"]), *conf["gauss"]["sigmas"]
            )

        # todo: add normalization option
        layer[layer > 255] = 255
        return layer

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
            sample_rate = self._config["m"]["sampling"]
            if sample_rate:
                if index % sample_rate != 0:
                    continue

            if index >= 1:
                fps = index / (time.time() - start_time)
            else:
                fps = 0

            self._logger.info(
                "Processing frame {:04} (fps: {:02.2f})".format(index, fps)
            )

            # ** calculations **

            diff = self.calc_diff(mean, frame)
            metric = self.calc_metric(diff)
            layer = self.calc_layer(frame, metric)

            sum_metric += metric

            # * overlay *
            layer_strat = self._config["m"]["overlay"]["strategy"]
            if layer_strat == "overlay":
                if index == 0:
                    sum_layer = frame
                else:
                    sum_layer = (1 - metric) * sum_layer + layer
            elif layer_strat == "add":
                sum_layer += layer
            elif layer_strat == "overlaymean":
                sum_layer = (1 - metric) * mean + layer
            else:
                raise ValueError

            if "merge" in self.preview or "merge" in self.save:
                merge = self.calc_merge(sum_layer, sum_metric)

            # ** Save/Preview **

            allowed = [
                "frame",
                "diff",
                "metric",
                "layer",
                "sum_metric",
                "sum_layer",
                "merge",
                "final",
            ]

            for item in self.save:
                if item == "final":
                    continue
                if item not in allowed:
                    msg = (
                        "Invalid value for save: '{}'. "
                        "Skipping this for now.".format(item)
                    )
                    self._logger.error(msg)
                    continue
                self.save_image(locals()[item], item, index)

            for item in self.preview:
                if item == "final":
                    continue
                if item not in allowed:
                    msg = (
                        "Invalid value for preview: '{}'. "
                        "Skipping this for now.".format(item)
                    )
                    self._logger.error(msg)
                    continue
                self.preview_image(locals()[item], item)

        if "final" in self.save or "final" in self.preview:
            merge = self.calc_merge(sum_layer, sum_metric)
        if "final" in self.save:
            self.save_image(merge, "final")
        if "final" in self.preview:
            self.preview_image(merge, "final")
