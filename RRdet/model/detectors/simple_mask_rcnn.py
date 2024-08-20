# Copyright (c) OpenMMLab. All rights reserved.
from ..builder import DETECTORS
from .two_stage import TwoStageDetector


@DETECTORS.register_module()
class SimpleMaskRCNN(TwoStageDetector):
    """Implementation of SimpleMaskRCNN"""

    def __init__(self,
                 backbone,
                 rpn_head,
                 roi_head,
                 train_cfg,
                 test_cfg,
                 neck=None,
                 pretrained=None,
                 init_cfg=None,
                 use_custom_features = None):
        super(SimpleMaskRCNN, self).__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg,
            use_custom_features = use_custom_features,)
