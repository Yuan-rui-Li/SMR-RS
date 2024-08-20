# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta, abstractmethod

from ...base_modules import BaseModule

from ..builder import build_shared_head


class BaseMaskRoIHead(BaseModule, metaclass=ABCMeta):
    """Base class for RoIHeads."""

    def __init__(self,
                 bbox_roi_extractor=None,
                 bbox_head=None,
                 mask_roi_extractor=None,
                 mask_head=None,
                 shared_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None,
                 bbox_head_on=False):
        #自定义参数
        self.bbox_head_on=bbox_head_on

        super(BaseMaskRoIHead, self).__init__(init_cfg)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        if mask_head is not None:
            self.init_mask_head(mask_roi_extractor, mask_head)

        self.init_assigner_sampler()


    @property
    def with_mask(self):
        """bool: whether the RoI head contains a `mask_head`"""
        return hasattr(self, 'mask_head') and self.mask_head is not None

    @abstractmethod
    def init_mask_head(self):
        """Initialize ``mask_head``"""
        pass

    @abstractmethod
    def init_assigner_sampler(self):
        """Initialize assigner and sampler."""
        pass

    @abstractmethod
    def forward_train(self,
                      x,
                      img_meta,
                      proposal_list,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      **kwargs):
        """Forward function during training."""

    async def async_simple_test(self,
                                x,
                                proposal_list,
                                img_metas,
                                proposals=None,
                                rescale=False,
                                **kwargs):
        """Asynchronized test function."""
        raise NotImplementedError

    def simple_test(self,
                    x,
                    proposal_list,
                    img_meta,
                    proposals=None,
                    rescale=False,
                    **kwargs):
        """Test without augmentation."""

    def aug_test(self, x, proposal_list, img_metas, rescale=False, **kwargs):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
