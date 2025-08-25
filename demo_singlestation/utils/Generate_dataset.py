import os

import numpy as np
import segyio
import h5py
import torch
import torch.utils.data as data
from scipy import interpolate
from sklearn.linear_model import LinearRegression
from torchvision import transforms

from utils.SpecEnhanced import spec_enhance


# 从多个segy文件中读取数据
# 将segy数据中的道数据转换为numpy数组