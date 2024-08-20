# Copyright (c) OpenMMLab. All rights reserved.
from . import mlu, mps
from .scatter_gather import scatter, scatter_kwargs
from .utils import get_device
from .mlu import *

__all__ = ['mlu', 'mps', 'get_device', 'scatter', 'scatter_kwargs']
