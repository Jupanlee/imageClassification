import gc
import os
from pathlib import Path
import random
import sys

from tqdm import tqdm_notebook as tqdm
import numpy as np 
import pandas as pd 

import matplotlib.pyplot as plt
import seaborn as sns

from IPython.core.display import display, HTML

# --- plotly ---
from plotly import tools, subplots
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.express as px
import plotly.figure_factory as ff

# --- models ---
from sklearn import preprocessing
from sklearn.model_selection import KFold

import argparse
from distutils.util import strtobool

import torch
from ignite.contrib.handlers import ProgressBar
from ignite.engine import Events
from numpy.random.mtrand import RandomState
from torch.utils.data.dataloader import DataLoader
import albumentations


# --- setup ---
pd.set_option('max_columns', 50)

num_fold = 0    
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

IMAGE_RGB_MEAN = [0.485, 0.456, 0.406]
IMAGE_RGB_STD  = [0.229, 0.224, 0.225]

from torch.utils.data.dataset import Dataset
import numpy
class DatasetMixin(Dataset):

    def __init__(self, transform = None):
        self.transform = transform

    def __getitem__(self, index):
        """Returns an example or a sequence of examples."""
        if torch.is_tensor(index):
            index = index.tolist()
        if isinstance(index, slice):
            current, stop, step = index.indices(len(self))
            return [self.get_example_wrapper(i) for i in
                    six.moves.range(current, stop, step)]
        elif isinstance(index, list) or isinstance(index, numpy.ndarray):
            return [self.get_example_wrapper(i) for i in index]
        else:
            return self.get_example_wrapper(index)

    def __len__(self):
        """Returns the number of data points."""
        raise NotImplementedError

    def get_example_wrapper(self, i):
        """Wrapper of `get_example`, to apply `transform` if necessary"""
        example = self.get_example(i)
        if self.transform:
            example = self.transform(example)
        return example

    def get_example(self, i):
        """Returns the i-th example.

        Implementations should override it. It should raise :class:`IndexError`
        if the index is invalid.

        Args:
            i (int): The index of the example.

        Returns:
            The i-th example.

        """
        raise NotImplementedError

# update BengaliAIDataset
class BengaliAIDataset(Dataset):
    def __init__(self, images, labels=None, transform=None, indices=None):
        self.images = images
        self.labels = labels
        self.transform = transform
        if indices is None:
            indices = np.arange(len(images))
        self.indices = indices
        self.train = labels is not None

    def __getitem__(self, i):
        """Return i-th data"""
        i = self.indices[i]
        x = self.images[i]
        # Opposite white and black: background will be white and
        # for future Affine transformation
        # x = (255 - x).astype(np.float32) / 255.
        x = np.stack([x, x, x], axis=2)
        if self.transform:
            x = self.transform(image=x)
            x = x['image']
        x = x.transpose(2,0,1)

        if self.train:
            y = self.labels[i]
            return x, y
        else:
            return x

    def __len__(self):
        """return length of this dataset"""
        return len(self.indices)

import cv2
from skimage.transform import AffineTransform, warp

def affine_image(img):
    """

    Args:
        img: (h, w) or (1, h, w)

    Returns:
        img: (h, w)
    """
    # ch, h, w = img.shape
    # img = img / 255.
    if img.ndim == 3:
        img = img[0]

    # --- scale ---
    min_scale = 0.8
    max_scale = 1.2
    sx = np.random.uniform(min_scale, max_scale)
    sy = np.random.uniform(min_scale, max_scale)

    # --- rotation ---
    max_rot_angle = 7
    rot_angle = np.random.uniform(-max_rot_angle, max_rot_angle) * np.pi / 180.

    # --- shear ---
    max_shear_angle = 10
    shear_angle = np.random.uniform(-max_shear_angle, max_shear_angle) * np.pi / 180.

    # --- translation ---
    max_translation = 4
    tx = np.random.randint(-max_translation, max_translation)
    ty = np.random.randint(-max_translation, max_translation)

    tform = AffineTransform(scale=(sx, sy), rotation=rot_angle, shear=shear_angle,
                            translation=(tx, ty))
    transformed_image = warp(img, tform)
    assert transformed_image.ndim == 2
    return transformed_image


def crop_char_image(image, threshold=5./255.):
    assert image.ndim == 2
    is_black = image > threshold

    is_black_vertical = np.sum(is_black, axis=0) > 0
    is_black_horizontal = np.sum(is_black, axis=1) > 0
    left = np.argmax(is_black_horizontal)
    right = np.argmax(is_black_horizontal[::-1])
    top = np.argmax(is_black_vertical)
    bottom = np.argmax(is_black_vertical[::-1])
    height, width = image.shape
    cropped_image = image[left:height - right, top:width - bottom]
    return cropped_image


def resize(image, size=(128, 128)):
    return cv2.resize(image, size)

class Transform:
    def __init__(self, affine=True, crop=True, size=(64, 64),
                 normalize=True, train=True):
        self.affine = affine
        self.crop = crop
        self.size = size
        self.normalize = normalize
        self.train = train

    def __call__(self, example):
        if self.train:
            x, y = example
        else:
            x = example
        # --- Augmentation ---
        if self.affine:
            x = affine_image(x)

        # --- Train/Test common preprocessing ---
        if self.crop:
            x = crop_char_image(x, threshold=40. / 255.)
        if self.size is not None:
            x = resize(x, size=self.size)
        if self.normalize:
            x = (x.astype(np.float32) - 0.0692) / 0.2051
        if x.ndim == 2:
            x = x[None, :, :]
        x = x.astype(np.float32)
        if self.train:
            y = y.astype(np.int64)
            return x, y
        else:
            return x

import pretrainedmodels
from torch import nn
import torch.nn.functional as F
from torch.nn import Sequential
from typing import List
from torch.nn.parameter import Parameter
import math
from torch.nn import init

def residual_add(lhs, rhs):
    lhs_ch, rhs_ch = lhs.shape[1], rhs.shape[1]
    if lhs_ch < rhs_ch:
        out = lhs + rhs[:, :lhs_ch]
    elif lhs_ch > rhs_ch:
        out = torch.cat([lhs[:, :rhs_ch] + rhs, lhs[:, rhs_ch:]], dim=1)
    else:
        out = lhs + rhs
    return out

class LazyLoadModule(nn.Module):
    """Lazy buffer/parameter loading using load_state_dict_pre_hook

    Define all buffer/parameter in `_lazy_buffer_keys`/`_lazy_parameter_keys` and
    save buffer with `register_buffer`/`register_parameter`
    method, which can be outside of __init__ method.
    Then this module can load any shape of Tensor during de-serializing.

    Note that default value of lazy buffer is torch.Tensor([]), while lazy parameter is None.
    """
    _lazy_buffer_keys: List[str] = []     # It needs to be override to register lazy buffer
    _lazy_parameter_keys: List[str] = []  # It needs to be override to register lazy parameter

    def __init__(self):
        super(LazyLoadModule, self).__init__()
        for k in self._lazy_buffer_keys:
            self.register_buffer(k, torch.tensor([]))
        for k in self._lazy_parameter_keys:
            self.register_parameter(k, None)
        self._register_load_state_dict_pre_hook(self._hook)

    def _hook(self, state_dict, prefix, local_metadata, strict, missing_keys,
             unexpected_keys, error_msgs):
        for key in self._lazy_buffer_keys:
            self.register_buffer(key, state_dict[prefix + key])

        for key in self._lazy_parameter_keys:
            self.register_parameter(key, Parameter(state_dict[prefix + key]))

class LazyLinear(LazyLoadModule):
    """Linear module with lazy input inference

    `in_features` can be `None`, and it is determined at the first time of forward step dynamically.
    """

    __constants__ = ['bias', 'in_features', 'out_features']
    _lazy_parameter_keys = ['weight']

    def __init__(self, in_features, out_features, bias=True):
        super(LazyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

        if in_features is not None:
            self.weight = Parameter(torch.Tensor(out_features, in_features))
            self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        if self.weight is None:
            self.in_features = input.shape[-1]
            self.weight = Parameter(torch.Tensor(self.out_features, self.in_features))
            self.reset_parameters()

            # Need to send lazy defined parameter to device...
            self.to(input.device)
        return F.linear(input, self.weight, self.bias)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )

class LinearBlock(nn.Module):

    def __init__(self, in_features, out_features, bias=True,
                 use_bn=True, activation=F.relu, dropout_ratio=-1, residual=False,):
        super(LinearBlock, self).__init__()
        if in_features is None:
            self.linear = LazyLinear(in_features, out_features, bias=bias)
        else:
            self.linear = nn.Linear(in_features, out_features, bias=bias)
        if use_bn:
            self.bn = nn.BatchNorm1d(out_features)
        if dropout_ratio > 0.:
            self.dropout = nn.Dropout(p=dropout_ratio)
        else:
            self.dropout = None
        self.activation = activation
        self.use_bn = use_bn
        self.dropout_ratio = dropout_ratio
        self.residual = residual

    def __call__(self, x):
        h = self.linear(x)
        if self.use_bn:
            h = self.bn(h)
        if self.activation is not None:
            h = self.activation(h)
        if self.residual:
            h = residual_add(h, x)
        if self.dropout_ratio > 0:
            h = self.dropout(h)
        return h

class PretrainedCNN(nn.Module):
    def __init__(self, model_name='se_resnext101_32x4d',
                 in_channels=1, out_dim=10, use_bn=True,
                 pretrained='imagenet'):
        super(PretrainedCNN, self).__init__()
       # self.conv0 = nn.Conv2d(
       #     in_channels, 3, kernel_size=3, stride=1, padding=1, bias=True)
        self.base_model = pretrainedmodels.__dict__[model_name](pretrained=pretrained)
        activation = F.leaky_relu
        self.do_pooling = True
        if self.do_pooling:
            inch = self.base_model.last_linear.in_features
        else:
            inch = None
        hdim = 512
        lin1 = LinearBlock(inch, hdim, use_bn=use_bn, activation=activation, residual=False)
        lin2 = LinearBlock(hdim, out_dim, use_bn=use_bn, activation=None, residual=False)
        self.lin_layers = Sequential(lin1, lin2)

    def forward(self, x):
        #h = self.conv0(x)
        h = self.base_model.features(x)

        if self.do_pooling:
            h = torch.sum(h, dim=(-1, -2))
        else:
            # [128, 2048, 4, 4] when input is (128, 128)
            bs, ch, height, width = h.shape
            h = h.view(bs, ch*height*width)
        for layer in self.lin_layers:
            h = layer(h)
        return h

class RGB(nn.Module):
    def __init__(self,):
        super(RGB, self).__init__()
        self.register_buffer('mean', torch.zeros(1,3,1,1))
        self.register_buffer('std', torch.ones(1,3,1,1))
        self.mean.data = torch.FloatTensor(IMAGE_RGB_MEAN).view(self.mean.shape)
        self.std.data = torch.FloatTensor(IMAGE_RGB_STD).view(self.std.shape)

    def forward(self, x):
        x = (x-self.mean)/self.std
        return x

class ConvBn2d(nn.Module):

    def __init__(self, in_channel, out_channel, kernel_size=3, padding=1, dilation=1, stride=1, groups=1, is_bn=True):
        super(ConvBn2d, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=groups, bias=False)
        self.bn   = nn.BatchNorm2d(out_channel, eps=1e-5)

    def forward(self,x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class SqueezeExcite(nn.Module):
    def __init__(self, in_channel, reduction=4,):
        super(SqueezeExcite, self).__init__()

        self.fc1 = nn.Conv2d(in_channel, in_channel//reduction, kernel_size=1, padding=0)
        self.fc2 = nn.Conv2d(in_channel//reduction, in_channel, kernel_size=1, padding=0)

    def forward(self, x):
        s = F.adaptive_avg_pool2d(x,1)
        s = self.fc1(s)
        s = F.relu(s, inplace=True)
        s = self.fc2(s)
        x = x*torch.sigmoid(s)
        return x

# bottleneck type C
class SENextBottleneck(nn.Module):
    def __init__(self, in_channel, channel, out_channel, stride=1, group=32,
                 reduction=16, pool=None, is_shortcut=False):
        super(SENextBottleneck, self).__init__()

        self.conv_bn1 = ConvBn2d(in_channel,     channel[0], kernel_size=1, padding=0, stride=1)
        self.conv_bn2 = ConvBn2d(   channel[0],  channel[1], kernel_size=3, padding=1, stride=1, groups=group)
        self.conv_bn3 = ConvBn2d(   channel[1], out_channel, kernel_size=1, padding=0, stride=1)
        self.scale    = SqueezeExcite(out_channel, reduction)

        #---
        self.is_shortcut = is_shortcut
        self.stride = stride
        if is_shortcut:
            self.shortcut = ConvBn2d(in_channel, out_channel, kernel_size=1, padding=0, stride=1)

        if stride==2:
            if pool=='max' : self.pool = nn.MaxPool2d(kernel_size=2,stride=2)
            if pool=='avg' : self.pool = nn.AvgPool2d(kernel_size=2,stride=2)

    def forward(self, x):
        z = F.relu(self.conv_bn1(x),inplace=True)
        z = F.relu(self.conv_bn2(z),inplace=True)
        if self.stride==2:
            z = self.pool(z)

        z = self.scale(self.conv_bn3(z))
        if self.is_shortcut:
            if self.stride==2:
                x = F.avg_pool2d(x,2,2)  #avg_pool2d
            x = self.shortcut(x)

        z += x
        z = F.relu(z,inplace=True)
        return z

#resnext50_32x4d
class ResNext50(nn.Module):

    def __init__(self, num_class=1000):
        super(ResNext50, self).__init__()
        self.rgb = RGB()

        self.block0  = nn.Sequential(
            # nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3, bias=False), #bias=0
            # nn.BatchNorm2d(64),
            # nn.ReLU(inplace=True),
            #nn.MaxPool2d(kernel_size=3, padding=0, stride=2, ceil_mode=True),
            #nn.MaxPool2d(kernel_size=2, padding=0, stride=2),
            #Identity(),

            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False), #bias=0
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False), #bias=0
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False), #bias=0
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.block1  = nn.Sequential(
             SENextBottleneck( 64, [128,128], 256, stride=2, is_shortcut=True, pool='max', ),
          * [SENextBottleneck(256, [128,128], 256, stride=1, is_shortcut=False,) for i in range(1,3)],
        )
        self.block2  = nn.Sequential(
             SENextBottleneck(256, [256,256], 512, stride=2, is_shortcut=True, pool='max', ),
          * [SENextBottleneck(512, [256,256], 512, stride=1, is_shortcut=False,) for i in range(1,4)],
        )
        self.block3  = nn.Sequential(
             SENextBottleneck( 512,[512,512],1024, stride=2, is_shortcut=True, pool='max', ),
          * [SENextBottleneck(1024,[512,512],1024, stride=1, is_shortcut=False,) for i in range(1,6)],
        )
        self.block4 = nn.Sequential(
             SENextBottleneck(1024,[1024,1024],2048, stride=2, is_shortcut=True,pool='avg', ),
          * [SENextBottleneck(2048,[1024,1024],2048, stride=1, is_shortcut=False) for i in range(1,3)],
        )

        self.logit = nn.Linear(2048, num_class)

    def forward(self, x):
        batch_size = len(x)
        #x = self.rgb(x)

        x = self.block0(x)
        #x = F.max_pool2d(x,kernel_size=2,stride=2)
        x = self.block1(x)
        #x = F.max_pool2d(x,kernel_size=2,stride=2)
        x = self.block2(x)
        #x = F.max_pool2d(x,kernel_size=2,stride=2)
        x = self.block3(x)
        #x = F.max_pool2d(x,kernel_size=2,stride=2)
        x = self.block4(x)
        x = F.adaptive_avg_pool2d(x,1).reshape(batch_size,-1)
        logit = self.logit(x)
        return logit

class Net(nn.Module):
    def __init__(self, num_class=10):
        super(Net, self).__init__()
        e = ResNext50()
        self.block0 = e.block0
        self.block1 = e.block1
        self.block2 = e.block2
        self.block3 = e.block3
        self.block4 = e.block4
        e = None

        self.dropblock0 = DropBlock2D(drop_prob=0.2, block_size=16)
        self.dropblock1 = DropBlock2D(drop_prob=0.2, block_size=8)

        self.logit = nn.ModuleList(
            [ nn.Linear(2048, c) for c in num_class ]
            )
    def forward(self, x):
        batch_size, C, H, W = x.shape
        if (H, W) != (64, 112):
            x = F.interpolate(x, size=(64, 112), mode='bilinear', aligh_corners=False)
        x = x.repeat(1, 3, 1, 1)
        x = self.block0(x)
        x = self.dropblock0(x)
        x = self.block1(x)
        x = self.dropblock1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        x = F.adaptive_avg_pool2d(x, 1).reshape(batch_size, -1)
        x = F.dropout(x, 0.2, self.training)

        # feature = None
        logit = [l(x) for l in self.logit]
        return logit

def accuracy(y, t):
    pred_label = torch.argmax(y, dim=1)
    count = pred_label.shape[0]
    correct = (pred_label == t).sum().type(torch.float32)
    acc = correct / count
    return acc
def onehot_encoding(label, n_classes):
    return torch.zeros(label.size(0), n_classes).to(label.device).scatter_(
        1, label.view(-1, 1), 1)
def cross_entropy_loss(input, target, reduction):
    logp = F.log_softmax(input, dim=1)
    loss = torch.sum(-logp * target, dim=1)
    if reduction == 'none':
        return loss
    elif reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        raise ValueError(
            '`reduction` must be one of \'none\', \'mean\', or \'sum\'.')
def label_smoothing_criterion(epsilon=0.1, reduction='mean'):
    def _label_smoothing_criterion(preds, targets):
        n_classes = preds.size(1)
        device = preds.device

        onehot = onehot_encoding(targets, n_classes).float().to(device)
        targets = onehot * (1 - epsilon) + torch.ones_like(onehot).to(
            device) * epsilon / n_classes
        loss = cross_entropy_loss(preds, targets, reduction)
        if reduction == 'none':
            return loss
        elif reduction == 'mean':
            return loss.mean()
        elif reduction == 'sum':
            return loss.sum()
        else:
            raise ValueError(
                '`reduction` must be one of \'none\', \'mean\', or \'sum\'.')

    return _label_smoothing_criterion

def mixup_criterion(preds1,preds2,preds3, targets):
    targets1, targets2,targets3, targets4,targets5, targets6, lam = targets[0], targets[1], targets[2], targets[3], targets[4], targets[5], targets[6]
    criterion = nn.CrossEntropyLoss(reduction='mean')
    return lam * criterion(preds1, targets1) + (1 - lam) * criterion(preds1, targets2) + lam * criterion(preds2, targets3) + (1 - lam) * criterion(preds2, targets4) + lam * criterion(preds3, targets5) + (1 - lam) * criterion(preds3, targets6)

def ohem_loss( rate, cls_pred, cls_target ):
    batch_size = cls_pred.size(0)
    ohem_cls_loss = F.cross_entropy(cls_pred, cls_target, reduction='none', ignore_index=-1)

    sorted_ohem_loss, idx = torch.sort(ohem_cls_loss, descending=True)
    keep_num = min(sorted_ohem_loss.size()[0], int(batch_size*rate) )
    if keep_num < sorted_ohem_loss.size()[0]:
        keep_idx_cuda = idx[:keep_num]
        ohem_cls_loss = ohem_cls_loss[keep_idx_cuda]
    cls_loss = ohem_cls_loss.sum() / keep_num
    return cls_loss

def cutmix_criterion(preds1,preds2,preds3, targets, rate=0.7):
    targets1, targets2,targets3, targets4,targets5, targets6, lam = targets[0], targets[1], targets[2], targets[3], targets[4], targets[5], targets[6]
    # criterion = nn.CrossEntropyLoss(reduction='mean')
    criterion = ohem_loss
    return lam * criterion(rate, preds1, targets1) + (1 - lam) * criterion(rate, preds1, targets2) + lam * criterion(rate, preds2, targets3) + (1 - lam) * criterion(rate, preds2, targets4) + lam * criterion(rate, preds3, targets5) + (1 - lam) * criterion(rate, preds3, targets6)

class BengaliClassifier(nn.Module):
    def __init__(self, predictor, n_grapheme=168, n_vowel=11, n_consonant=7):
        super(BengaliClassifier, self).__init__()
        self.n_grapheme = n_grapheme
        self.n_vowel = n_vowel
        self.n_consonant = n_consonant
        self.n_total_class = self.n_grapheme + self.n_vowel + self.n_consonant
        self.predictor = predictor

#         self.metrics_keys = [
#             'loss', 'loss_grapheme', 'loss_vowel', 'loss_consonant',
#             'acc_grapheme', 'acc_vowel', 'acc_consonant']
        self.metrics_keys = ['loss']

    def forward(self, x, y=None):
        pred = self.predictor(x)
        if isinstance(pred, tuple):
            assert len(pred) == 3
            preds = pred
        else:
            assert pred.shape[1] == self.n_total_class
            preds = torch.split(pred, [self.n_grapheme, self.n_vowel, self.n_consonant], dim=1)
#         loss_grapheme = F.cross_entropy(preds[0], y[:, 0])
#         loss_vowel = F.cross_entropy(preds[1], y[:, 1])
#         loss_consonant = F.cross_entropy(preds[2], y[:, 2])
#         loss = loss_grapheme + loss_vowel + loss_consonant
        
        loss = cutmix_criterion(preds[0], preds[1], preds[2], y)
#         acc_grapheme = accuracy(preds[0], y[:, 0])
#         acc_vowel = accuracy(preds[1], y[:, 1])
#         acc_consonant = accuracy(preds[2], y[:, 2])
        
        metrics = {
            'loss': loss.item()
#             'loss_grapheme': loss_grapheme.item(),
#             'loss_vowel': loss_vowel.item(),
#             'loss_consonant': loss_consonant.item(),
#             'acc_grapheme': acc_grapheme.item(),
#             'acc_vowel': acc_vowel.item(),
#             'acc_consonant': acc_consonant.item()
        }
        return loss, metrics, pred

    def calc(self, data_loader):
        device: torch.device = next(self.parameters()).device
        self.eval()
        output_list = []
        with torch.no_grad():
            for batch in tqdm(data_loader):
                # TODO: support general preprocessing.
                # If `data` is not `Data` instance, `to` method is not supported!
                batch = batch.to(device)
                pred = self.predictor(batch)
                output_list.append(pred)
        output = torch.cat(output_list, dim=0)
        preds = torch.split(output, [self.n_grapheme, self.n_vowel, self.n_consonant], dim=1)
        return preds

    def predict_proba(self, data_loader):
        preds = self.calc(data_loader)
        return [F.softmax(p, dim=1) for p in preds]

    def predict(self, data_loader):
        preds = self.calc(data_loader)
        pred_labels = [torch.argmax(p, dim=1) for p in preds]
        return pred_labels

debug = False
submission = False
batch_size = 256 
device = 'cuda:0'
out = '.'
image_size = 128
arch = 'pretrained'
model_name = 'resnet34'

datadir = Path('/home/ehl/lijupan/kaggle/input/bengaliai-cv19')
featherdir = Path('/home/ehl/lijupan/kaggle/input/bengaliaicv19feather')
outdir = Path(f'./data_test/{model_name}')
if not os.path.exists(outdir):
    os.makedirs(outdir)

# --- Model ---
device = torch.device(device)
n_grapheme = 168
n_vowel = 11
n_consonant = 7
n_total = n_grapheme + n_vowel + n_consonant
print('n_total', n_total)
# predictor = PretrainedCNN(in_channels=1, out_dim=n_total, model_name=model_name, pretrained=None)
predictor = Net(num_class=n_total)
print('predictor', type(predictor))

classifier = BengaliClassifier(predictor).to(device)

from logging import getLogger
from time import perf_counter
import json

# from chainer_chemistry.utils import save_json

from ignite.engine.engine import Engine, Events
from ignite.metrics import Average

def save_json(filepath, params):
    with open(filepath, 'w') as f:
        json.dump(params, f, indent=4)

class DictOutputTransform:
    def __init__(self, key, index=0):
        self.key = key
        self.index = index

    def __call__(self, x):
        if self.index >= 0:
            x = x[self.index]
        return x[self.key]

def mixup(data, target, targets1, targets2, targets3, alpha):
    indices = torch.randperm(data.size(0))
    shuffled_data = data[indices]
    shuffled_target = target[indices]
    shuffled_targets1 = targets1[indices]
    shuffled_targets2 = targets2[indices]
    shuffled_targets3 = targets3[indices]

    lam = np.random.beta(alpha, alpha)
    data = data * lam + shuffled_data * (1 - lam)
    targets = [targets1, shuffled_targets1, targets2, shuffled_targets2, targets3, shuffled_targets3, lam]

    return data, targets, shuffled_target

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def cutmix(data, target, targets1, targets2, targets3, alpha):
    indices = torch.randperm(data.size(0))
    shuffled_data = data[indices]
    shuffled_target = target[indices]
    shuffled_targets1 = targets1[indices]
    shuffled_targets2 = targets2[indices]
    shuffled_targets3 = targets3[indices]

    lam = np.random.beta(alpha, alpha)
    bbx1, bby1, bbx2, bby2 = rand_bbox(data.size(), lam)
    data[:, :, bbx1:bbx2, bby1:bby2] = data[indices, :, bbx1:bbx2, bby1:bby2]
    # adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (data.size()[-1] * data.size()[-2]))

    targets = [targets1, shuffled_targets1, targets2, shuffled_targets2, targets3, shuffled_targets3, lam]
    return data, targets, shuffled_target

def valid_mixup(data, target, targets1, targets2, targets3, alpha):
    lam = np.random.beta(alpha, alpha)
    targets = [targets1, targets1, targets2, targets2, targets3, targets3, lam]
    return data, targets, target

def create_trainer(classifier, optimizer, device):
    classifier.to(device)

    def update_fn(engine, batch):
        classifier.train()
        optimizer.zero_grad()
        # batch = [elem.to(device) for elem in batch]
        x, y = [elem.to(device) for elem in batch]
        x, y, y_old = cutmix(x, y, y[:,0], y[:,1], y[:,2], alpha=1.0)
        loss, metrics, pred_y = classifier(x, y)
        loss.backward()
        optimizer.step()
        return metrics, pred_y, y_old
    trainer = Engine(update_fn)

    for key in classifier.metrics_keys:
        Average(output_transform=DictOutputTransform(key)).attach(trainer, key)
    return trainer

def create_evaluator(classifier, device):
    classifier.to(device)

    def update_fn(engine, batch):
        classifier.eval()
        with torch.no_grad():
            # batch = [elem.to(device) for elem in batch]
            x, y = [elem.to(device) for elem in batch]
            x, y, y_old = valid_mixup(x, y, y[:,0], y[:,1], y[:,2], alpha=1.0)
            _, metrics, pred_y = classifier(x, y)
            return metrics, pred_y, y_old
    evaluator = Engine(update_fn)

    for key in classifier.metrics_keys:
        Average(output_transform=DictOutputTransform(key)).attach(evaluator, key)
    return evaluator

class trainLogReport:
    def __init__(self, dirpath=None, logger=None):
        self.dirpath = str(dirpath) if dirpath is not None else None
        self.logger = logger or getLogger(__name__)

        self.reported_dict = {}  # To handle additional parameter to monitor
        self.history = []
        self.start_time = perf_counter()

    def report(self, key, value):
        self.reported_dict[key] = value

    def __call__(self, engine):
        elapsed_time = perf_counter() - self.start_time
        elem = {'epoch': engine.state.epoch,
                'iteration': engine.state.iteration}
        losses = engine.state.output[0].items()
        elem.update({f'train/{key}': value for key, value in losses})
        elem.update(self.reported_dict)
        elem['elapsed_time'] = elapsed_time
        self.history.append(elem)

        if self.dirpath:
            save_json(os.path.join(self.dirpath, 'train_log.json'), self.history)
            self.get_dataframe().to_csv(os.path.join(self.dirpath, 'train_log.csv'), index=False)

        # --- print ---
        msg = ''
        for key, value in elem.items():
            if isinstance(value, int):
                msg += f'{key} {value: >6d} '
            else:
                msg += f'{key} {value: 8f} '

        print(msg)

        # --- Reset ---
        self.reported_dict = {}

    def get_dataframe(self):
        df = pd.DataFrame(self.history)
        return df

class validLogReport:
    def __init__(self, evaluator=None, dirpath=None, logger=None):
        self.evaluator = evaluator
        self.dirpath = str(dirpath) if dirpath is not None else None
        self.logger = logger or getLogger(__name__)

        self.reported_dict = {}  # To handle additional parameter to monitor
        self.history = []
        self.start_time = perf_counter()

    def report(self, key, value):
        self.reported_dict[key] = value

    def __call__(self, engine):
        elapsed_time = perf_counter() - self.start_time
        elem = {'epoch': engine.state.epoch,
                'iteration': engine.state.iteration}

        if self.evaluator is not None:
            if self.evaluator.state is not None:
                elem.update({f'valid/{key}': value
                             for key, value in self.evaluator.state.metrics.items()})

        elem.update(self.reported_dict)
        elem['elapsed_time'] = elapsed_time
        self.history.append(elem)

        if self.dirpath:
            save_json(os.path.join(self.dirpath, 'valid_log.json'), self.history)
            self.get_dataframe().to_csv(os.path.join(self.dirpath, 'valid_log.csv'), index=False)

        # --- print ---
        msg = ''
        for key, value in elem.items():
#             if key in ['iteration']:
#                 # skip printing some parameters...
#                 continue
#             elif isinstance(value, int):
            if isinstance(value, int):
                msg += f'{key} {value: >6d} '
            else:
                msg += f'{key} {value: 8f} '
#         self.logger.warning(msg)
        print(msg)

        # --- Reset ---
        self.reported_dict = {}

    def get_dataframe(self):
        df = pd.DataFrame(self.history)
        return df

class SpeedCheckHandler:
    def __init__(self, iteration_interval=10, logger=None):
        self.iteration_interval = iteration_interval
        self.logger = logger or getLogger(__name__)
        self.prev_time = perf_counter()

    def __call__(self, engine: Engine):
        if engine.state.iteration % self.iteration_interval == 0:
            cur_time = perf_counter()
            spd = self.iteration_interval / (cur_time - self.prev_time)
            self.logger.warning(f'{spd} iter/sec')
            # reset
            self.prev_time = cur_time

    def attach(self, engine: Engine):
        engine.add_event_handler(Events.ITERATION_COMPLETED, self)


class ModelSnapshotHandler:
    def __init__(self, model, fold, filepath, filename='model_{count:06}.pt',
                 interval=1, logger=None):
        self.model = model
        self.fold = fold
        self.filepath: str = str(filepath)
        self.filename = filename
        self.interval = interval
        self.logger = logger or getLogger(__name__)
        self.count = 0

    def __call__(self, engine: Engine):
        self.count += 1
        if self.count % self.interval == 0:
            filename = 'fold'+str(self.fold)+'_'+self.filename.format(count=self.count)
            torch.save(self.model.state_dict(), os.path.join(self.filepath, filename))
            # self.logger.warning(f'save model to {filepath}...')

import warnings
from ignite.metrics.metric import Metric
import sklearn.metrics


class EpochMetric(Metric):
    """Class for metrics that should be computed on the entire output history of a model.
    Model's output and targets are restricted to be of shape `(batch_size, n_classes)`. Output
    datatype should be `float32`. Target datatype should be `long`.

    .. warning::

        Current implementation stores all input data (output and target) in as tensors before computing a metric.
        This can potentially lead to a memory error if the input data is larger than available RAM.


    - `update` must receive output of the form `(y_pred, y)`.

    If target shape is `(batch_size, n_classes)` and `n_classes > 1` than it should be binary: e.g. `[[0, 1, 0, 1], ]`.

    Args:
        compute_fn (callable): a callable with the signature (`torch.tensor`, `torch.tensor`) takes as the input
            `predictions` and `targets` and returns a scalar.
        output_transform (callable, optional): a callable that is used to transform the
            :class:`~ignite.engine.Engine`'s `process_function`'s output into the
            form expected by the metric. This can be useful if, for example, you have a multi-output model and
            you want to compute the metric with respect to one of the outputs.

    """

    def __init__(self, compute_fn, output_transform=lambda x: x):

        if not callable(compute_fn):
            raise TypeError("Argument compute_fn should be callable.")

        super(EpochMetric, self).__init__(output_transform=output_transform)
        self.compute_fn = compute_fn

    def reset(self):
        self._predictions = torch.tensor([], dtype=torch.float32)
        self._targets = torch.tensor([], dtype=torch.long)

    def update(self, output):
        y_pred, y = output
        self._predictions = torch.cat([self._predictions, y_pred], dim=0)
        self._targets = torch.cat([self._targets, y], dim=0)

        # Check once the signature and execution of compute_fn
        if self._predictions.shape == y_pred.shape:
            try:
                self.compute_fn(self._predictions, self._targets)
            except Exception as e:
                warnings.warn("Probably, there can be a problem with `compute_fn`:\n {}.".format(e),
                              RuntimeWarning)

    def compute(self):
        return self.compute_fn(self._predictions, self._targets)

def macro_recall(pred_y, y, n_grapheme=168, n_vowel=11, n_consonant=7):
    pred_y = torch.split(pred_y, [n_grapheme, n_vowel, n_consonant], dim=1)
    pred_labels = [torch.argmax(py, dim=1).cpu().numpy() for py in pred_y]

    y = y.cpu().numpy()
    # pred_y = [p.cpu().numpy() for p in pred_y]

    recall_grapheme = sklearn.metrics.recall_score(pred_labels[0], y[:, 0], average='macro')
    recall_vowel = sklearn.metrics.recall_score(pred_labels[1], y[:, 1], average='macro')
    recall_consonant = sklearn.metrics.recall_score(pred_labels[2], y[:, 2], average='macro')
    scores = [recall_grapheme, recall_vowel, recall_consonant]
    final_score = np.average(scores, weights=[2, 1, 1])
    # print(f'recall: grapheme {recall_grapheme}, vowel {recall_vowel}, consonant {recall_consonant}, '
    #       f'total {final_score}, y {y.shape}')
    return final_score


def calc_macro_recall(solution, submission):
    # solution df, submission df
    scores = []
    for component in ['grapheme_root', 'consonant_diacritic', 'vowel_diacritic']:
        y_true_subset = solution[solution[component] == component]['target'].values
        y_pred_subset = submission[submission[component] == component]['target'].values
        scores.append(sklearn.metrics.recall_score(
            y_true_subset, y_pred_subset, average='macro'))
    final_score = np.average(scores, weights=[2, 1, 1])
    return final_score


def prepare_image(datadir, featherdir, data_type='train',
                  submission=False, indices=[0, 1, 2, 3]):
    assert data_type in ['train', 'test']
    if submission:
        image_df_list = [pd.read_parquet(datadir / f'{data_type}_image_data_{i}.parquet', engine='pyarrow')
                         for i in indices]
    else:
        image_df_list = [pd.read_feather(featherdir / f'{data_type}_image_data_{i}.feather')
                         for i in indices]

#     print('image_df_list', len(image_df_list))
    HEIGHT = 137
    WIDTH = 236
    ids = [df.iloc[:, 0] for df in image_df_list]
    ids = pd.concat(ids, axis=0)
    images = [df.iloc[:, 1:].values.reshape(-1, HEIGHT, WIDTH) for df in image_df_list]
    images = np.concatenate(images, axis=0)
    print(images.shape)
    
    return ids, images

def output_transform(output):
    metric, pred_y, y = output
    return pred_y.cpu(), y.cpu()

def run_evaluator(engine):
    evaluator.run(valid_loader)
    
def schedule_lr(engine):
    # metrics = evaluator.state.metrics
    metrics = engine.state.metrics
    avg_mae = metrics['loss']

    # --- update lr ---
    lr = scheduler.optimizer.param_groups[0]['lr']
    scheduler.step(avg_mae)
    valid_log_report.report('lr', lr)

BASE_LR = 0.001
    
train_csv = pd.read_csv(datadir/'train.csv')

# prepare cross validation
fold_train_indices = [[1,2,3], [0,2,3], [0,1,3], [0,1,2]]
#NUM_FOLD = 4 
c_train_ids, c_train_images = prepare_image(datadir, featherdir, data_type='train',
                                            submission=True, indices=fold_train_indices[num_fold])
c_train_labels = c_train_ids.to_frame().join(train_csv.set_index('image_id'), on='image_id')[['grapheme_root', 'vowel_diacritic', 'consonant_diacritic']].values

c_val_ids, c_val_images = prepare_image(datadir, featherdir, data_type='train',
                                        submission=True, indices=[num_fold])
c_val_labels = c_val_ids.to_frame().join(train_csv.set_index('image_id'), on='image_id')[['grapheme_root', 'vowel_diacritic', 'consonant_diacritic']].values
#     print(c_train_ids)
#     print(c_val_ids)

train_transform = albumentations.Compose([
    albumentations.Resize(image_size, image_size),
    albumentations.Normalize(mean=0.456, std=0.224, max_pixel_value=255.0, p=1.0)
])
    
val_transform = albumentations.Compose([
    albumentations.Resize(image_size, image_size),
    albumentations.Normalize(mean=0.456, std=0.224, max_pixel_value=255.0, p=1.0)
])

train_dataset = BengaliAIDataset(c_train_images, c_train_labels, 
                                 transform=train_transform)
val_dataset = BengaliAIDataset(c_val_images, c_val_labels, 
                               transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

optimizer = torch.optim.Adam(classifier.parameters(), lr = BASE_LR)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=5, min_lr=1e-10)

trainer = create_trainer(classifier, optimizer, device)
EpochMetric(compute_fn = macro_recall, output_transform = output_transform).attach(trainer, 'recall')

pbar = ProgressBar()
pbar.attach(trainer, metric_names='all')

evaluator = create_evaluator(classifier, device)
EpochMetric(compute_fn = macro_recall, output_transform = output_transform).attach(evaluator, 'recall')

trainer.add_event_handler(Events.EPOCH_COMPLETED, run_evaluator)
trainer.add_event_handler(Events.EPOCH_COMPLETED, schedule_lr)
train_log_report = trainLogReport(outdir)
valid_log_report = validLogReport(evaluator, outdir)
trainer.add_event_handler(Events.ITERATION_COMPLETED(every=100), train_log_report)
trainer.add_event_handler(Events.EPOCH_COMPLETED, valid_log_report)

# model_save_path = outdir / (str(num_fold)+'predictor.pt')

trainer.add_event_handler(
    Events.EPOCH_COMPLETED,
    ModelSnapshotHandler(predictor, fold=num_fold, filepath=outdir))

trainer.run(train_loader, max_epochs=100)
