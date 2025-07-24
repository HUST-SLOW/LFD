import sys
import os
import logging
import re
import functools
import fnmatch
import numpy as np
import cv2
import torch
import torchvision.transforms.functional as F
import numbers
import random 
import warnings
from sklearn.metrics._ranking import _binary_clf_curve
from sklearn.exceptions import UndefinedMetricWarning



def setup_logger(distributed_rank=0, filename="log.txt"):
    """Set up the logger.
    Args:
        distributed_rank (int): The rank of the current process in distributed training.
        filename (str): The name of the log file.
    Returns:
        logging.Logger: The logger.
    """
    logger = logging.getLogger("Logger")
    logger.setLevel(logging.DEBUG)
    # don't log results for the non-master process
    if distributed_rank > 0:
        return logger
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
    ch.setFormatter(logging.Formatter(fmt))
    logger.addHandler(ch)

    return logger


class AverageMeter(object):
    """Computes and stores the average and current value of a meter.
    Args:
        name (str): The name of the meter.
    Attributes:
        initialized (bool): Indicates whether the meter has been initialized.
        val (float): The current value of the meter.
        avg (float): The average value of the meter.
        sum (float): The sum of all values multiplied by their respective weights.
        count (float): The total weight of all values.
    Methods:
        initialize(val, weight): Initializes the meter with a value and weight.
        update(val, weight=1): Updates the meter with a new value and weight.
        add(val, weight): Adds a value to the meter with a given weight.
        value(): Returns the current value of the meter.
        average(): Returns the average value of the meter.
    """
    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = val * weight
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum += val * weight
        self.count += weight
        self.avg = self.sum / self.count

    def value(self):
        return self.val

    def average(self):
        return self.avg


def normalization01(img):
    """Normalize the input image to the range [0, 1].
    Args:
        img (numpy.ndarray): The input image.
    Returns:
        numpy.ndarray: The normalized image.
    """
    return (img - img.min()) / (img.max() - img.min())


def denormalization(x):
    """Denormalize the input image.
    Args:
        x (numpy.ndarray): The input image.
    Returns:
        numpy.ndarray: The denormalized image.
    """
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    x = (((x * std) + mean) * 255.).astype(np.uint8)
    return x


def iou_curve(gt, scores, pos_label=None, sample_weight=None,
              drop_intermediate=True):
    """Compute per-region overlap-false positive rate pairs for different probability thresholds.
    Args:
        gt (numpy.ndarray): Ground truth of the classifier, 1-d numpy array binary mask [0,1].
        scores (numpy.ndarray): Classifier scores.
    Returns:
        numpy.ndarray: False positive rate.
        numpy.ndarray: Per-region overlap.
        numpy.ndarray: Thresholds.
    """
    assert gt.shape == scores.shape

    fps, tps, thresholds = _binary_clf_curve(
        gt, scores, pos_label=pos_label, sample_weight=sample_weight)

    if drop_intermediate and len(fps) > 2:
        optimal_idxs = np.where(np.r_[True,
                                      np.logical_or(np.diff(fps, 2),
                                                    np.diff(tps, 2)),
                                      True])[0]
        fps = fps[optimal_idxs]
        tps = tps[optimal_idxs]
        thresholds = thresholds[optimal_idxs]

    if tps.size == 0 or fps[0] != 0:
        # Add an extra threshold position if necessary
        tps = np.r_[0, tps]
        fps = np.r_[0, fps]
        thresholds = np.r_[thresholds[0] + 1, thresholds]

    if fps[-1] <= 0:
        warnings.warn("No negative samples in y_true, "
                      "false positive value should be meaningless",
                      UndefinedMetricWarning)
        fpr = np.repeat(np.nan, fps.shape)
    else:
        fpr = fps / fps[-1]

    if tps[-1] <= 0:
        warnings.warn("No positive samples in y_true, "
                      "true positive value should be meaningless",
                      UndefinedMetricWarning)
        tpr = np.repeat(np.nan, tps.shape)
    else:
        tpr = tps / tps[-1]

    fns = tps[-1] - tps
    # compute per-region overlap from confusion-matrix
    iou = tps / (tps + fps + fns)

    # PRO curve up to an average false-positive rate of 30%
    index_range = np.where(fpr <= 1.0)
    index_range = index_range[0]
    fpr = fpr[index_range]
    iou = iou[index_range]
    thresholds = thresholds[index_range]

    return fpr, iou, thresholds


def accuracy(preds, label):
    """Calculate the accuracy of the predictions.
    Args:
        preds (numpy.ndarray): The predicted labels.
        label (numpy.ndarray): The true labels.
    Returns:
        float: The accuracy of the predictions.
        int: The number of valid samples.
    """
    valid = (label >= 0)
    acc_sum = (valid * (preds == label)).sum()
    valid_sum = valid.sum()
    acc = float(acc_sum) / (valid_sum + 1e-10)
    return acc, valid_sum


def intersectionAndUnion(imPred, imLab, numClass):
    """Calculate the intersection and union areas between predicted and labeled images.
    Args:
        imPred (numpy.ndarray): The predicted image.
        imLab (numpy.ndarray): The labeled image.
        numClass (int): The number of classes.
    Returns:
        tuple: A tuple containing the area of intersection and the area of union.
    """
    imPred = np.asarray(imPred).copy()
    imLab = np.asarray(imLab).copy()

    imPred += 1
    imLab += 1
    # Remove classes from unlabeled pixels in gt image.
    # We should not penalize detections in unlabeled portions of the image.
    imPred = imPred * (imLab > 0)

    # Compute area intersection:
    intersection = imPred * (imPred == imLab)
    (area_intersection, _) = np.histogram(
        intersection, bins=numClass, range=(1, numClass))

    # Compute area union:
    (area_pred, _) = np.histogram(imPred, bins=numClass, range=(1, numClass))
    (area_lab, _) = np.histogram(imLab, bins=numClass, range=(1, numClass))
    area_union = area_pred + area_lab - area_intersection

    return (area_intersection, area_union)


class NotSupportedCliException(Exception):
    pass


def process_range(xpu, inp):
    """Process the range of devices.
    Args:
        xpu (str): The device type.
        inp (tuple): The range of devices.
    Returns:
        list: The list of devices.
    """
    start, end = map(int, inp)
    if start > end:
        end, start = start, end
    return map(lambda x: '{}{}'.format(xpu, x), range(start, end+1))


REGEX = [
    (re.compile(r'^gpu(\d+)$'), lambda x: ['gpu%s' % x[0]]),
    (re.compile(r'^(\d+)$'), lambda x: ['gpu%s' % x[0]]),
    (re.compile(r'^gpu(\d+)-(?:gpu)?(\d+)$'),
     functools.partial(process_range, 'gpu')),
    (re.compile(r'^(\d+)-(\d+)$'),
     functools.partial(process_range, 'gpu')),
]


def parse_devices(input_devices):
    """Parse user's devices input string to standard format.
    Args:
        input_devices (str): User's devices input string.    
    Returns:
        list: List of parsed devices in standard format.    
    Raises:
        NotSupportedCliException: If the device input cannot be recognized.
    """
    ret = []
    for d in input_devices.split(','):
        for regex, func in REGEX:
            m = regex.match(d.lower().strip())
            if m:
                tmp = func(m.groups())
                # prevent duplicate
                for x in tmp:
                    if x not in ret:
                        ret.append(x)
                break
        else:
            raise NotSupportedCliException(
                'Cannot recognize device: "{}"'.format(d))
    return ret


class ExtRandomCrop(object):
    """Crop the given PIL Image at a random location.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
        padding (int or sequence, optional): Optional padding on each border
            of the image. Default is 0, i.e no padding. If a sequence of length
            4 is provided, it is used to pad left, top, right, bottom borders
            respectively.
        pad_if_needed (boolean): It will pad the image if smaller than the
            desired size to avoid raising an exception.
    """

    def __init__(self, size, padding=0, pad_if_needed=False):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding
        self.pad_if_needed = pad_if_needed

    @staticmethod
    def get_params(img, output_size):
        """Get parameters for ``crop`` for a random crop.
        Args:
            img (PIL Image): Image to be cropped.
            output_size (tuple): Expected output size of the crop.
        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
        """
        w, h = img.size
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    def __call__(self, img, lbl, mask):
        """
        Args:
            img (PIL Image): Image to be cropped.
            lbl (PIL Image): Label to be cropped.
        Returns:
            PIL Image: Cropped image.
            PIL Image: Cropped label.
        """
        assert img.size == lbl.size, 'size of img and lbl should be the same. %s, %s'%(img.size, lbl.size)
        if self.padding > 0:
            img = F.pad(img, self.padding)
            lbl = F.pad(lbl, self.padding)
            mask = F.pad(mask, self.padding)

        # pad the width if needed
        if self.pad_if_needed and img.size[0] < self.size[1]:
            img = F.pad(img, padding=int((1 + self.size[1] - img.size[0]) / 2))
            lbl = F.pad(lbl, padding=int((1 + self.size[1] - lbl.size[0]) / 2),)
            mask = F.pad(mask, padding=int((1 + self.size[1] - mask.size[0]) / 2))

        # pad the height if needed
        if self.pad_if_needed and img.size[1] < self.size[0]:
            img = F.pad(img, padding=int((1 + self.size[0] - img.size[1]) / 2))
            lbl = F.pad(lbl, padding=int((1 + self.size[0] - lbl.size[1]) / 2))
            mask = F.pad(mask, padding=int((1 + self.size[0] - mask.size[1]) / 2))

        i, j, h, w = self.get_params(img, self.size)

        return F.crop(img, i, j, h, w), F.crop(lbl, i, j, h, w), F.crop(mask, i, j, h, w)


def setup_seed(seed):
    """Set the global random seed to ensure the reproducibility of the experiment.
    Args:
        seed (int): The random seed to set.
    Returns:
        None.
    """
    torch.manual_seed(seed)  # Set the random seed for PyTorch's CPU
    torch.cuda.manual_seed(seed)  # Set the random seed for PyTorch's current GPU
    torch.cuda.manual_seed_all(seed)  # Set the random seed for all GPUs in PyTorch
    np.random.seed(seed)  # Set the random seed for NumPy
    random.seed(seed)  # Set the random seed for Python's built-in random number generator
    os.environ["PYTHONHASHSEED"] = str(seed)  # Set the seed for Python's hash algorithm
    torch.backends.cudnn.deterministic = True  # Set the CuDNN's algorithm to be deterministic
    torch.backends.cudnn.benchmark = False  
    torch.use_deterministic_algorithms(True, warn_only=True)