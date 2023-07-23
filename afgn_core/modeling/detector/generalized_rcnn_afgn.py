# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Implements the Generalized R-CNN framework
"""
import time

from PIL import Image
from collections import deque

import torch

import torchvision
from torch import nn


from afgn_core.structures.image_list import to_image_list
from afgn_core.structures.boxlist_ops import cat_boxlist
from afgn_core.structures.boxlist_ops  import boxlist_iou


import numpy as np
import os
import random
import cv2

from ..backbone import build_backbone
from ..rpn.rpn import build_rpn
from ..roi_heads.roi_heads import build_roi_heads
from .se_net import se_block
from .cbam_net import eca_block

from skimage.metrics import structural_similarity as ssim

from torchvision.transforms import Resize 






class GeneralizedRCNNAFGN(nn.Module):
    """
    Main class for Generalized R-CNN. Currently supports boxes and masks.
    It consists of three main parts:
    - backbone
    - rpn
    - heads: takes the features + the proposals from the RPN and computes
        detections / masks from it.
    """
    
    def __init__(self, cfg):
        super(GeneralizedRCNNAFGN, self).__init__()
        self.device = cfg.MODEL.DEVICE

        self.backbone = build_backbone(cfg)
        

        self.rpn = build_rpn(cfg, self.backbone.out_channels)
        self.roi_heads = build_roi_heads(cfg, self.backbone.out_channels)
        self.key_frame_location = cfg.MODEL.VID.AFGN.KEY_FRAME_LOCATION
        # self.feat1_att   = eca_block(self.backbone.out_channels)


    def prepare_onnx_export(self):
        self.rpn.prepare_onnx_export()
        self.roi_heads.prepare_onnx_export()
        
    def forward(self, images, targets=None):
        """
        Arguments:
            #images (list[Tensor] or ImageList): images to be processed
            targets (list[BoxList]): ground-truth boxes present in the image (optional)
        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).
        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")

        if self.training:
            images["cur"] = to_image_list(images["cur"])
            images["ref"] = [to_image_list(image) for image in images["ref"]]

            return self._forward_train(images["cur"], images["ref"], images["ref_id"], targets)
        else:
            images["cur"] = to_image_list(images["cur"])
            images["ref"] = [to_image_list(image) for image in images["ref"]]

            infos = images.copy()
            infos.pop("cur")
            return self._forward_test(images["cur"], infos)
    


    def Selecting(self, proposal_lsit, img_refs_id):
        cur_proposal_list = proposal_lsit[0][0]
        ref_proposal_list1 = proposal_lsit[1]
        ref_proposal_list2 = proposal_lsit[2]
        score1 = ref_proposal_list1.get_field("objectness")
        score2 = ref_proposal_list2.get_field("objectness")
        iou_result1 = boxlist_iou(ref_proposal_list1,cur_proposal_list)
        iou_result2 = boxlist_iou(ref_proposal_list2,cur_proposal_list)
        iou1 = []
        iou2 = []
        # print(len(ref_proposal_list1))
        # print(iou_result1.shape)
        for i in range(len(ref_proposal_list1)):
            iou1.append(torch.sum(iou_result1[i])/torch.sum(iou_result1))
        for j in range(len(ref_proposal_list2)):
            iou2.append(torch.sum(iou_result2[j])/torch.sum(iou_result2))
    
        avg_score1 = []
        avg_score2 = []

        for k in range(len(ref_proposal_list1)):
            avg_score1.append((score1[k] + (2*5-abs(img_refs_id[0]))/(2*5) + iou1[k] )/3)
        for k in range(len(ref_proposal_list2)):
            avg_score2.append((score2[k] + (2*5-abs(img_refs_id[1]))/(2*5) + iou2[k] )/3)
        
        if len(ref_proposal_list1)>75:
            _, topk_idx1 = torch.topk(torch.tensor(avg_score1), 75, dim=0, sorted=True)
            boxlists1 = ref_proposal_list1[topk_idx1]

        if len(ref_proposal_list2)>75:
            _, topk_idx2 = torch.topk(torch.tensor(avg_score2), 75, dim=0, sorted=True)
            boxlists2 = ref_proposal_list2[topk_idx2]
        

        proposals_list = []
        proposals_list.append([cur_proposal_list])
        proposals_list.append(boxlists1)
        proposals_list.append(boxlists2)


        # _, topk_idx1 = torch.topk(torch.tensor(avg_score1), 25, dim=0, sorted=True)
        # boxlists11 = ref_proposal_list1[topk_idx1]
        # _, topk_idx2 = torch.topk(torch.tensor(avg_score2), 25, dim=0, sorted=True)
        # boxlists22 = ref_proposal_list2[topk_idx2]
        # _, topk_idx = torch.topk(cur_proposal_list.get_field("objectness"), 75, dim=0, sorted=True)
        # cur = cur_proposal_list[topk_idx]

        # self.hua(cur,boxlists1,boxlists2,boxlists11,boxlists22)

        return proposals_list

    def _forward_train(self, img_cur, imgs_ref, img_refs_id, targets):
       
     
        concat_imgs = torch.cat([img_cur.tensors, *[img_ref.tensors for img_ref in imgs_ref]], dim=0)
        concat_feats = self.backbone(concat_imgs)[0]
    
        num_imgs = 1 + len(imgs_ref)

        feats_list = torch.chunk(concat_feats, num_imgs, dim=0)

        # propagate the features
        proposals_list = []
        # key frame
        proposals, proposal_losses = self.rpn(img_cur, (feats_list[0],), targets, version="key")
        proposals_list.append(proposals)


        for i in range(len(imgs_ref)):
            proposals_ref = self.rpn(imgs_ref[i], (feats_list[i + 1], ), version="ref")
            proposals_list.append(proposals_ref[0])

        proposals_list = self.Selecting(proposals_list, img_refs_id)

        if self.roi_heads:
            x, result, detector_losses = self.roi_heads(feats_list,
                                                        proposals_list,
                                                        targets)
        else:
            detector_losses = {}

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)

        return losses

    def _forward_test(self, imgs, infos, targets=None):
        """
        forward for the test phase.
        :param imgs:
        :param frame_category: 0 for start, 1 for normal
        :param targets:
        :return:
        """

        def update_feature(img=None, feats=None, proposals=None, proposals_feat=None):
            assert (img is not None) or (feats is not None and proposals is not None and proposals_feat is not None)

            if img is not None:
                feats = self.backbone(img)[0]
                # feats=self.feat1_att(feats)
                # note here it is `imgs`! for we only need its shape, it would not cause error, but is not explicit.
                proposals = self.rpn(imgs, (feats,), version="ref")
                proposals_feat = self.roi_heads.box.feature_extractor(feats, proposals, pre_calculate=True)

            self.feats.append(feats)
            self.proposals.append(proposals[0])
            self.proposals_feat.append(proposals_feat)


        if targets is not None:
            raise ValueError("In testing mode, targets should be None")

        # if infos["frame_category"] == '0':  # a new video
        if infos["frame_category"] == 0:  # a new video
            self.seg_len = infos["seg_len"]
            self.end_id = 0
 
            self.feats = deque(maxlen=30)
            self.proposals = deque(maxlen=30)
            self.proposals_feat = deque(maxlen=30)



            feats_cur = self.backbone(imgs.tensors)[0]
        
    
           
            proposals_cur = self.rpn(imgs, (feats_cur, ), version="ref")
            proposals_feat_cur = self.roi_heads.box.feature_extractor(feats_cur, proposals_cur, pre_calculate=True)
            while len(self.feats) < self.key_frame_location + 1:
                update_feature(None, feats_cur, proposals_cur, proposals_feat_cur)

            while len(self.feats) < 30:
                self.end_id = min(self.end_id + 1, self.seg_len - 1)
                end_filename = infos["pattern"] % self.end_id
                end_image = Image.open(infos["img_dir"] % end_filename).convert("RGB")

                end_image = infos["transforms"](end_image)
                if isinstance(end_image, tuple):
                    end_image = end_image[0]
                end_image = end_image.view(1, *end_image.shape).to(self.device)

                update_feature(end_image)

        elif infos["frame_category"] == 1:
            self.end_id = min(self.end_id + 1, self.seg_len - 1)
            end_image = infos["ref"][0].tensors

            update_feature(end_image)

        feats = self.feats[self.key_frame_location]
        proposals, proposal_losses = self.rpn(imgs, (feats, ), None)

        proposals_ref = cat_boxlist(list(self.proposals))
        proposals_feat_ref = torch.cat(list(self.proposals_feat), dim=0)


        proposals_list = [proposals, proposals_ref, proposals_feat_ref]
        if self.roi_heads:
            x, result, detector_losses = self.roi_heads(feats, proposals_list, None)
        else:
            result = proposals

        return result


