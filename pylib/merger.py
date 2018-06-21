#!/usr/bin/env python3

from . import log
import numpy as np
import os.path
import os
import cv2


class Merger(object):
    """ This class overlays frames and produces an output image. """
    def __init__(self, inpt):

        self._input = inpt

        self.name = os.path.splitext(os.path.basename(inpt.path))[0]

        self.image_format = "png"

        self._logger = log.setup_logger("Merger")

        # which steps should be saved
        self.save = []
        # live preview
        self.preview = []

        # For convenience:
        self._shape_rgb = inpt.shape  # height x width x 3
        self._shape_scalar = (self._shape_rgb[0], self._shape_rgb[1], 1)
        self._shape = (self._shape_rgb[0], self._shape_rgb[1])

    @property
    def output_dir(self):
        return os.path.join("out", self.name)

    def run(self):
        raise NotImplementedError

    def get_final_image(self):
        raise NotImplementedError

    def preview_image(self, image, name="image", max_height=500, max_width=None):

        # ** determine size to be displayed **
        old_height, old_width, _ = image.shape
        if max_width and max_height:
            new_size = (max_height, max_width)
        elif max_width:
            if max_width >= old_width:
                new_size = (old_height, old_width)
            else:
                scale = max_width/old_width
                new_size = (scale * old_height, scale * old_width)
        elif max_height:
            if max_height >= old_height:
                new_size = (old_height, old_width)
            else:
                scale = max_height/old_height
                new_size = (scale * old_height, scale * old_width)
        else:
            new_size = (old_height, old_width)
        new_size = (int(new_size[0]), int(new_size[1]))

        # ** resize **
        resized = cv2.resize(image, (new_size[1], new_size[0])).astype(np.uint8)

        # ** display **
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


class SimpleMerger(Merger):
    """ This merger first calculates a mean (default: average over all frames),
    then, for each frame calculates a difference (default: mean - difference),
    from which we calculate a metric, (e.g. R^3 distance).
    We then sum over all frame * metric and divide by the sum of metric. """

    def __init__(self, inpt):
        """ Initialize
        :param inpt: InputData object (from pylib/input)
        """
        super().__init__(inpt)

        # Progress images to save
        # mean, diff, metric, merge, final
        self.save = ["final"]

        # these values will be continuously updated in the self.run loop
        # This will allow us more flexibility than using parameters for the
        # class functions.

        self.index = 0      # number of current frame
        self.frame = None   # current frame
        self.mean = None    # mean of all frames
        self.diff = None    # mean - frame
        self.metric = None  # metric(diff)

        self.sum_metric = np.zeros(shape=self._shape_scalar, dtype=float)

        # We call metric * frame a layer. This is the sum of those
        self.sum_layers = np.zeros(shape=self._shape_rgb, dtype=float)

        # This is sum_layers/sum_metric. Also continuously updated if we
        # choose to save it.
        self.final = None

    def calc_mean(self):
        self.mean = sum(self._input.get_frames()) / self._input.number_images

    def calc_diff(self):
        self.diff = self.mean - self.frame

    def calc_metric(self):
        raise NotImplementedError

    def calc_sum_metric(self):
        self.sum_metric += self.metric

    def calc_sum_layers(self):
        self.sum_layers += self.frame * self.metric

    def calc_final(self):
        self.final = self.sum_layers / self.sum_metric

    def calc_all(self):
        self.calc_diff()
        self.calc_metric()
        self.calc_sum_metric()
        self.calc_sum_layers()
        if "merge" in self.preview or "merge" in self.save:
            self.calc_final()

    def run(self):
        self._logger.debug("Run!")

        self.calc_mean()
        if "mean" in self.save:
            self.save_image(self.mean, "mean")
        if "mean" in self.preview:
            self.preview_image(self.mean, "mean")

        for self.index, self.frame in enumerate(self._input.get_frames()):
            self.calc_all()

            if "frame" in self.save:
                self.save_image(self.frame, "frame", self.index)
            if "diff" in self.save:
                self.save_image(self.diff, "diff", self.index)
            if "metric" in self.save:
                self.save_image(self.scalar_to_grayscale(self.metric),
                                "metric",
                                self.index)
            if "merge" in self.save:
                self.save_image(self.final, "merge", self.index)

            if "frame" in self.preview:
                self.preview_image(self.frame, "frame")
            if "diff" in self.preview:
                self.preview_image(self.diff, "diff")
            if "metric" in self.preview:
                self.preview_image(self.scalar_to_grayscale(self.metric), "metric")
            if "merge" in self.preview:
                self.preview_image(self.final, "merge")

        self.calc_final()
        if "final" in self.save:
            self.save_image(self.final, "final")
        if "final" in self.preview:
            self.preview_image(self.final, "final")

    def get_final_image(self):
        return self.final


class R3Merger(SimpleMerger):

    def __init__(self, inpt):
        super().__init__(inpt)

        self.intensity = 500
        self.blur_shape = (5, 5)

    def calc_metric(self):
        # change intensity to increase emphasis of outliers
        metric = 1 + self.intensity * np.sqrt(np.sum(np.square(self.diff/255), axis=-1))
        metric = cv2.GaussianBlur(metric, self.blur_shape, 0)
        self.metric = metric.reshape(self._shape_scalar)


class SimpleMeanMerger(R3Merger):

    def __init__(self, inpt):
        super().__init__(inpt)

    def calc_mean(self):
        for frame in self._input.get_frames():
            self.mean = frame
            return


class CutoffMerger(SimpleMerger):

    def __init__(self, inpt):
        super().__init__(inpt)
        self.metric_threshold = 0.1
        self.metric_min = 0.1 / self._input.number_images
        self.metric_max = 1
        self.metric_blur_shape = (5, 5)
        self.metric_blur_sigma = 0

    def calc_metric(self):
        metric = np.sqrt(np.sum(np.square(self.diff/255), axis=-1))
        metric = cv2.GaussianBlur(metric, self.metric_blur_shape, self.metric_blur_sigma)
        metric = np.piecewise(
            metric,
            [metric < self.metric_threshold, metric >= self.metric_threshold],
            [self.metric_min, self.metric_max])
        self.metric = metric.reshape(self._shape_scalar)


class SimpleMeanCutoffMerger(CutoffMerger):

    def __init__(self, inpt):
        super().__init__(inpt)

    def calc_mean(self):
        for frame in self._input.get_frames():
            self.mean = frame
            return


class PatchedMeanCutoffMerger(CutoffMerger):

    def __init__(self, inpt):
        super().__init__(inpt)

    def calc_mean(self):
        width = self._input.shape[1]
        width_left = int(width/2)
        left = self._input.get_frame(-1)[:, :width_left, :]
        right = self._input.get_frame(0)[:, width_left:, :]
        self.mean = np.concatenate((left, right), axis=1)


class OverlayMerger(PatchedMeanCutoffMerger):

    def __init__(self, inpt):
        super().__init__(inpt)
        self.metric_min = 0

    def calc_sum_layers(self):
        if self.index == 0:
            self.sum_layers = self.frame
        else:
            self.sum_layers = (1-self.metric) * self.sum_layers + \
                              self.metric * self.frame

    def calc_final(self):
        self.final = self.sum_layers


class RunningDifferenceMerger(SimpleMeanCutoffMerger):

    def __init__(self, inpt):
        super().__init__(inpt)
        self.metric_threshold = 0.05

    def calc_diff(self):
        super().calc_diff()
        self.diff[self.diff < 0] = 0

    def calc_all(self):
        super().calc_all()
        self.mean = self.frame


class EdgeDetectionMerger(PatchedMeanCutoffMerger):

    def __init__(self, inpt):
        super().__init__(inpt)
        self.metric_min = 0
        self.metric_blur_shape = (11, 11)
        self.metric_blur_sigma = 5

    def calc_metric(self):
        super().calc_metric()
        gray = self.metric * 255
        gray = gray.reshape(self._shape).astype(np.uint8)
        edges = cv2.Canny(gray, 100, 200)
        edges = edges.reshape(self._shape_scalar).astype(np.float)
        self.metric = edges

    def calc_sum_layers(self):
        if self.index == 0:
            # todo: if I use self.mean here, something really funny happens. Why?
            self.sum_layers = self.frame
        else:
            self.sum_layers += self.metric

    def calc_final(self):
        self.final = self.sum_layers


# class FifoMerger(object):
#     """ This class overlays frames and produces an output image. """
#     def __init__(self, inpt, fifo_length=1):
#
#         self.input = inpt
#
#         # height x width x 3
#         self.shape_rgb = inpt.shape
#         # height x width x 1
#         self.shape_scalar = (self.shape_rgb[0], self.shape_rgb[1], 1)
#
#         self.logger = log.setup_logger("merger")
#
#         # currently processed frame
#         self.frame_no = 0
#
#         # FIFOs = First in first out: Keep copies of the last n frames
#         self.fifo_frame = collections.deque(maxlen=fifo_length)
#         self.fifo_mean = collections.deque(maxlen=fifo_length)
#         self.fifo_diff = collections.deque(maxlen=fifo_length)
#         self.fifo_metric = collections.deque(maxlen=fifo_length)
#         self.fifo_merged = collections.deque(maxlen=fifo_length)
#
#
#         # Sums
#         self.sum_metric = np.zeros(self.shape_scalar)
#         self.sum_layers = np.zeros(self.shape_scalar)
#
#         # Config
#         self.save_diff = False
#         self.save_mean = True
#         self.save_metric = False
