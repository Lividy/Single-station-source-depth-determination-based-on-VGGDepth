
""" Additional utility functions. """
import os
import time
import pprint
import torch
import numpy as np
import torch.nn.functional as F
from torchvision import transforms

class Averager():
    """The class to calculate the average."""
    def __init__(self):
        self.n = 0
        self.v = 0

    def add(self, x):
        self.v = (self.v * self.n + x) / (self.n + 1)
        self.n += 1

    def item(self):
        return self.v

def count_acc(logits, label):
    """The function to calculate the .
    Args:
      logits: input logits.
      label: ground truth labels.
    Return:
      The output accuracy.
    """
    pred = F.softmax(logits, dim=1).argmax(dim=1)
    if torch.cuda.is_available():
        return (pred == label).type(torch.cuda.FloatTensor).mean().item()
    return (pred == label).type(torch.FloatTensor).mean().item()

def Resize_Feature(feature, resize_n):    
    feature = transforms.ToPILImage()(feature)
    resize_feature = transforms.Resize(size=resize_n)(feature)
    tensor_feature = transforms.ToTensor()(resize_feature)
    return tensor_feature.numpy()

class Timer():
    """The class for timer."""
    def __init__(self):
        self.o = time.time()

    def measure(self, p=1):
        x = (time.time() - self.o) / p
        x = int(x)
        if x >= 3600:
            return '{:.1f}h'.format(x / 3600)
        if x >= 60:
            return '{}m'.format(round(x / 60))
        return '{}s'.format(x)

_utils_pp = pprint.PrettyPrinter()

def pprint(x):
    _utils_pp.pprint(x)

def compute_confidence_interval(data):
    """The function to calculate the .
    Args:
      data: input records
      label: ground truth labels.
    Return:
      m: mean value
      pm: confidence interval.
    """
    a = 1.0 * np.array(data)
    m = np.mean(a)
    std = np.std(a)
    pm = 1.96 * (std / np.sqrt(len(a)))
    return m, pm

def set_gpu(cuda_device):
    os.environ['CUDA_VISIBLE_DEVICES'] = cuda_device
    
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    print('Using gpu:', cuda_device)

