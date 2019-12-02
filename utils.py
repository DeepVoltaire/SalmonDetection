from pathlib import Path
import sys
from selenium import webdriver
from selenium.webdriver.firefox.options import Options
import psutil
import datetime
from time import gmtime, strftime
import matplotlib.image as mpimg
import os, time
import pandas as pd
import numpy as np
import torch
import random
import geopandas as gpd
from shapely.geometry import Point, Polygon
import shutil
from torch.utils.data import Dataset, Sampler
import albumentations
import logging
import tifffile
import glob
import matplotlib.pyplot as plt
import cv2
from sklearn.metrics import log_loss
from tqdm import tqdm
import seaborn as sns
sns.set_style("white")
import operator
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from PIL import Image
from crawling import *
from train import *

LONG_PER_KM = 0.0118  # 0.0118 long at this latitude is roughly 1 km
LAT_PER_KM = 0.009  # 0.009 lat is roughly 1 km