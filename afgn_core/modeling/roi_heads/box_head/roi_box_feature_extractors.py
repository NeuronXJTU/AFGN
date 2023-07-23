# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import math
from collections import deque

import torch
from torch import nn
from torch.nn import functional as F

from afgn_core.modeling import registry
from afgn_core.modeling.backbone import resnet
from afgn_core.modeling.poolers import Pooler
from afgn_core.modeling.make_layers import group_norm
from afgn_core.modeling.make_layers import make_fc, Conv2d

from afgn_core.structures.boxlist_ops import cat_boxlist

@registry.ROI_BOX_FEATURE_EXTRACTORS.register("ResNet50Conv5ROIFeatureExtractor")
class ResNet50Conv5ROIFeatureExtractor(nn.Module):
    def __init__(self, config, in_channels):
        super(ResNet50Conv5ROIFeatureExtractor, self).__init__()

        resolution = config.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        scales = config.MODEL.ROI_BOX_HEAD.POOLER_SCALES
        sampling_ratio = config.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler = Pooler(
            output_size=(resolution, resolution),
            scales=scales,
            sampling_ratio=sampling_ratio,
        )

        stage = resnet.StageSpec(index=4, block_count=3, return_features=False)
        head = resnet.ResNetHead(
            block_module=config.MODEL.RESNETS.TRANS_FUNC,
            stages=(stage,),
            num_groups=config.MODEL.RESNETS.NUM_GROUPS,
            width_per_group=config.MODEL.RESNETS.WIDTH_PER_GROUP,
            stride_in_1x1=config.MODEL.RESNETS.STRIDE_IN_1X1,
            stride_init=None,
            res2_out_channels=config.MODEL.RESNETS.RES2_OUT_CHANNELS,
            dilation=config.MODEL.RESNETS.RES5_DILATION
        )

        self.pooler = pooler
        self.head = head
        self.out_channels = head.out_channels


    def forward(self, x, proposals):
        x = self.pooler(x, proposals)
        x = self.head(x)
        return x


@registry.ROI_BOX_FEATURE_EXTRACTORS.register("ResNetConv52MLPFeatureExtractor")
class ResNetConv52MLPFeatureExtractor(nn.Module):
    """
    Heads for Faster R-CNN MSRA version for classification
    """

    def __init__(self, cfg, in_channels):
        super(ResNetConv52MLPFeatureExtractor, self).__init__()

        stage = resnet.StageSpec(index=4, block_count=3, return_features=False)
        head = resnet.ResNetHead(
            block_module=cfg.MODEL.RESNETS.TRANS_FUNC,
            stages=(stage,),
            num_groups=cfg.MODEL.RESNETS.NUM_GROUPS,
            width_per_group=cfg.MODEL.RESNETS.WIDTH_PER_GROUP,
            stride_in_1x1=cfg.MODEL.RESNETS.STRIDE_IN_1X1,
            stride_init=1,
            res2_out_channels=cfg.MODEL.RESNETS.RES2_OUT_CHANNELS,
            dilation=cfg.MODEL.RESNETS.RES5_DILATION,
        )

        in_channels = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS * 2 ** (stage.index - 1)
        if cfg.MODEL.VID.ROI_BOX_HEAD.REDUCE_CHANNEL:
            new_conv = nn.Conv2d(in_channels, 256, kernel_size=1, stride=1)
            nn.init.kaiming_uniform_(new_conv.weight, a=1)
            nn.init.constant_(new_conv.bias, 0)
            output_channel = 256
        else:
            new_conv = None
            output_channel = in_channels

        resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        scales = cfg.MODEL.ROI_BOX_HEAD.POOLER_SCALES
        sampling_ratio = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler = Pooler(
            output_size=(resolution, resolution),
            scales=scales,
            sampling_ratio=sampling_ratio,
        )

        self.head = head
        self.conv = new_conv
        self.pooler = pooler

        input_size = output_channel * resolution ** 2
        representation_size = cfg.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM
        use_gn = cfg.MODEL.ROI_BOX_HEAD.USE_GN
        self.fc6 = make_fc(input_size, representation_size, use_gn)
        self.fc7 = make_fc(representation_size, representation_size, use_gn)

        self.out_channels = representation_size

    def forward(self, x, proposals):
        if self.conv is not None:
            x = self.head(x[0])
            x = (F.relu(self.conv(x)), )
        else:
            x = (self.head(x[0]), )
        x = self.pooler(x, proposals)

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))

        return x


class AttentionExtractor(nn.Module):
    def __init__(self, cfg, in_channels):
        super(AttentionExtractor, self).__init__()

    @staticmethod
    def extract_position_embedding(position_mat, feat_dim, wave_length=1000.0):
        device = position_mat.device
        # position_mat, [num_rois, num_nongt_rois, 4]
        feat_range = torch.arange(0, feat_dim / 8, device=device)
        dim_mat = torch.full((len(feat_range),), wave_length, device=device).pow(8.0 / feat_dim * feat_range)
        dim_mat = dim_mat.view(1, 1, 1, -1).expand(*position_mat.shape, -1)

        position_mat = position_mat.unsqueeze(3).expand(-1, -1, -1, dim_mat.shape[3])
        position_mat = position_mat * 100.0

        div_mat = position_mat / dim_mat
        sin_mat, cos_mat = div_mat.sin(), div_mat.cos()

        # [num_rois, num_nongt_rois, 4, feat_dim / 4]
        embedding = torch.cat([sin_mat, cos_mat], dim=3)
        # [num_rois, num_nongt_rois, feat_dim]
        embedding = embedding.reshape(embedding.shape[0], embedding.shape[1], embedding.shape[2] * embedding.shape[3])

        return embedding

    @staticmethod
    def extract_position_matrix(bbox, ref_bbox):
        xmin, ymin, xmax, ymax = torch.chunk(ref_bbox, 4, dim=1)
        bbox_width_ref = xmax - xmin + 1
        bbox_height_ref = ymax - ymin + 1
        center_x_ref = 0.5 * (xmin + xmax)
        center_y_ref = 0.5 * (ymin + ymax)

        xmin, ymin, xmax, ymax = torch.chunk(bbox, 4, dim=1)
        bbox_width = xmax - xmin + 1
        bbox_height = ymax - ymin + 1
        center_x = 0.5 * (xmin + xmax)
        center_y = 0.5 * (ymin + ymax)

        delta_x = center_x - center_x_ref.transpose(0, 1)
        delta_x = delta_x / bbox_width
        delta_x = (delta_x.abs() + 1e-3).log()

        delta_y = center_y - center_y_ref.transpose(0, 1)
        delta_y = delta_y / bbox_height
        delta_y = (delta_y.abs() + 1e-3).log()

        delta_width = bbox_width / bbox_width_ref.transpose(0, 1)
        delta_width = delta_width.log()

        delta_height = bbox_height / bbox_height_ref.transpose(0, 1)
        delta_height = delta_height.log()

        position_matrix = torch.stack([delta_x, delta_y, delta_width, delta_height], dim=2)

        return position_matrix

    def attention_module_multi_head(self, roi_feat, ref_feat, position_embedding,
                                    feat_dim=1024, dim=(1024, 1024, 1024), group=16,
                                    index=0):
        """

        :param roi_feat: [num_rois, feat_dim]
        :param ref_feat: [num_nongt_rois, feat_dim]
        :param position_embedding: [1, emb_dim, num_rois, num_nongt_rois]
        :param feat_dim: should be same as dim[2]
        :param dim: a 3-tuple of (query, key, output)
        :param group:
        :return:
        """
        dim_group = (dim[0] / group, dim[1] / group, dim[2] / group)

        # position_embedding, [1, emb_dim, num_rois, num_nongt_rois]
        # -> position_feat_1, [1, group, num_rois, num_nongt_rois]
        position_feat_1 = F.relu(self.Wgs[index](position_embedding))
        # aff_weight, [num_rois, group, num_nongt_rois, 1]
        aff_weight = position_feat_1.permute(2, 1, 3, 0)
        # aff_weight, [num_rois, group, num_nongt_rois]
        aff_weight = aff_weight.squeeze(3)

        # multi head
        assert dim[0] == dim[1]

        q_data = self.Wqs[index](roi_feat)
        q_data_batch = q_data.reshape(-1, group, int(dim_group[0]))
        # q_data_batch, [group, num_rois, dim_group[0]]
        q_data_batch = q_data_batch.permute(1, 0, 2)

        k_data = self.Wks[index](ref_feat)
        k_data_batch = k_data.reshape(-1, group, int(dim_group[1]))
        # k_data_batch, [group, num_nongt_rois, dim_group[1]]
        k_data_batch = k_data_batch.permute(1, 0, 2)

        # v_data, [num_nongt_rois, feat_dim]
        v_data = ref_feat

        # aff, [group, num_rois, num_nongt_rois]
        aff = torch.bmm(q_data_batch, k_data_batch.transpose(1, 2))
        aff_scale = (1.0 / math.sqrt(float(dim_group[1]))) * aff
        # aff_scale, [num_rois, group, num_nongt_rois]
        aff_scale = aff_scale.permute(1, 0, 2)

        # weighted_aff, [num_rois, group, num_nongt_rois]
        weighted_aff = (aff_weight + 1e-6).log() + aff_scale
        aff_softmax = F.softmax(weighted_aff, dim=2)

        aff_softmax_reshape = aff_softmax.reshape(aff_softmax.shape[0] * aff_softmax.shape[1], aff_softmax.shape[2])

        # output_t, [num_rois * group, feat_dim]
        output_t = torch.matmul(aff_softmax_reshape, v_data)
        # output_t, [num_rois, group * feat_dim, 1, 1]
        output_t = output_t.reshape(-1, group * feat_dim, 1, 1)
        # linear_out, [num_rois, dim[2], 1, 1]
        linear_out = self.Wvs[index](output_t)

        output = linear_out.squeeze(3).squeeze(2)

        return output

    def cal_position_embedding(self, rois1, rois2):
        # [num_rois, num_nongt_rois, 4]
        position_matrix = self.extract_position_matrix(rois1, rois2)
        # [num_rois, num_nongt_rois, 64]
        position_embedding = self.extract_position_embedding(position_matrix, feat_dim=64)
        # [64, num_rois, num_nongt_rois]
        position_embedding = position_embedding.permute(2, 0, 1)
        # [1, 64, num_rois, num_nongt_rois]
        position_embedding = position_embedding.unsqueeze(0)

        return position_embedding


@registry.ROI_BOX_FEATURE_EXTRACTORS.register("AFGNFeatureExtractor")
class AFGNFeatureExtractor(AttentionExtractor):
    """
    Heads for Faster R-CNN MSRA version for classification
    """

    def __init__(self, cfg, in_channels):
        super(AFGNFeatureExtractor, self).__init__(cfg, in_channels)

        stage = resnet.StageSpec(index=4, block_count=3, return_features=False)
        head = resnet.ResNetHead(
            block_module=cfg.MODEL.RESNETS.TRANS_FUNC,
            stages=(stage,),
            num_groups=cfg.MODEL.RESNETS.NUM_GROUPS,
            width_per_group=cfg.MODEL.RESNETS.WIDTH_PER_GROUP,
            stride_in_1x1=cfg.MODEL.RESNETS.STRIDE_IN_1X1,
            stride_init=1,
            res2_out_channels=cfg.MODEL.RESNETS.RES2_OUT_CHANNELS,
            dilation=cfg.MODEL.RESNETS.RES5_DILATION,
        )

        in_channels = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS * 2 ** (stage.index - 1)
        if cfg.MODEL.VID.ROI_BOX_HEAD.REDUCE_CHANNEL:
            new_conv = nn.Conv2d(in_channels, 256, kernel_size=1, stride=1)
            nn.init.kaiming_uniform_(new_conv.weight, a=1)
            nn.init.constant_(new_conv.bias, 0)
            output_channel = 256
        else:
            new_conv = None
            output_channel = in_channels

        resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        scales = cfg.MODEL.ROI_BOX_HEAD.POOLER_SCALES
        sampling_ratio = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler = Pooler(
            output_size=(resolution, resolution),
            scales=scales,
            sampling_ratio=sampling_ratio,
        )

        self.head = head
        self.conv = new_conv
        self.pooler = pooler

        input_size = output_channel * resolution ** 2
        representation_size = cfg.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM
        use_gn = cfg.MODEL.ROI_BOX_HEAD.USE_GN

        if cfg.MODEL.VID.ROI_BOX_HEAD.ATTENTION.ENABLE:
            self.embed_dim = cfg.MODEL.VID.ROI_BOX_HEAD.ATTENTION.EMBED_DIM
            self.groups = cfg.MODEL.VID.ROI_BOX_HEAD.ATTENTION.GROUP
            self.feat_dim = representation_size

            self.stage = cfg.MODEL.VID.ROI_BOX_HEAD.ATTENTION.STAGE


            fcs, Wgs, Wqs, Wks, Wvs = [], [], [], [], []

            for i in range(self.stage):
                r_size = input_size if i == 0 else representation_size
                fcs.append(make_fc(r_size, representation_size, use_gn))
                Wgs.append(Conv2d(self.embed_dim, self.groups, kernel_size=1, stride=1, padding=0))
                Wqs.append(make_fc(self.feat_dim, self.feat_dim))
                Wks.append(make_fc(self.feat_dim, self.feat_dim))
                Wvs.append(Conv2d(self.feat_dim * self.groups, self.feat_dim, kernel_size=1, stride=1, padding=0, groups=self.groups))
                for l in [Wgs[i], Wvs[i]]:
                    torch.nn.init.normal_(l.weight, std=0.01)
                    torch.nn.init.constant_(l.bias, 0)
            self.fcs = nn.ModuleList(fcs)
            self.Wgs = nn.ModuleList(Wgs)
            self.Wqs = nn.ModuleList(Wqs)
            self.Wks = nn.ModuleList(Wks)
            self.Wvs = nn.ModuleList(Wvs)

        self.out_channels = representation_size
        
    def forward(self, x, proposals, pre_calculate=False):
        if pre_calculate:
            return self._forward_ref(x, proposals)

        if self.training:
            return self._forward_train(x, proposals)
        else:
            return self._forward_test(x, proposals)

    def _forward_train(self, x, proposals):
        num_refs = len(x) - 1
        x = self.head(torch.cat(x, dim=0))
        if self.conv is not None:
            x = F.relu(self.conv(x))
        x, x_refs = torch.split(x, [1, num_refs], dim=0)


        proposals, proposals_refs1, proposals_refs2 = proposals[0][0], proposals[1], proposals[2:]

        x, x_cur = torch.split(self.pooler((x, ), [cat_boxlist([proposals, proposals_refs1], ignore_field=True), ]), [len(proposals), len(proposals_refs1)], dim=0)
    
        x, x_cur = x.flatten(start_dim=1), x_cur.flatten(start_dim=1)

        if proposals_refs2:
            x_refs = self.pooler((x_refs, ), proposals_refs2)
            x_refs = x_refs.flatten(start_dim=1)
            x_refs = torch.cat([x_cur, x_refs], dim=0)
        else:
            x_refs = x_cur

        rois_cur = proposals.bbox
    
        rois_ref = cat_boxlist([proposals_refs1, *proposals_refs2]).bbox
        position_embedding = self.cal_position_embedding(rois_cur, rois_ref)

        x_refs = F.relu(self.fcs[0](x_refs))

        for i in range(self.stage):
            x = F.relu(self.fcs[i](x))
            attention = self.attention_module_multi_head(x, x_refs, position_embedding,
                                                         feat_dim=1024, group=16, dim=(1024, 1024, 1024),
                                                         index=i)
            x = x + attention
        

        
        return x

    def _forward_ref(self, x, proposals):
        if self.conv is not None:
            x = self.head(x)
            x = (F.relu(self.conv(x)), )
        else:
            x = (self.head(x), )
        x = self.pooler(x, proposals)
        x = x.flatten(start_dim=1)

        x = F.relu(self.fcs[0](x))

        return x

    def _forward_test(self, x, proposals):
        proposals, proposals_ref, x_refs = proposals

        rois_cur = cat_boxlist(proposals).bbox
        rois_ref = proposals_ref.bbox

        if self.conv is not None:
            x = self.head(x)
            x = (F.relu(self.conv(x)), )
        else:
            x = (self.head(x), )
        x = self.pooler(x, proposals)
        x = x.flatten(start_dim=1)

        position_embedding = self.cal_position_embedding(rois_cur, rois_ref)

        for i in range(self.stage):
            x = F.relu(self.fcs[i](x))
            attention = self.attention_module_multi_head(x, x_refs, position_embedding,
                                                         feat_dim=1024, group=16, dim=(1024, 1024, 1024),
                                                         index=i)
            x = x + attention


        return x
    
    

@registry.ROI_BOX_FEATURE_EXTRACTORS.register("FPN2MLPFeatureExtractor")
class FPN2MLPFeatureExtractor(nn.Module):
    """
    Heads for FPN for classification
    """

    def __init__(self, cfg, in_channels):
        super(FPN2MLPFeatureExtractor, self).__init__()

        resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        scales = cfg.MODEL.ROI_BOX_HEAD.POOLER_SCALES
        sampling_ratio = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler = Pooler(
            output_size=(resolution, resolution),
            scales=scales,
            sampling_ratio=sampling_ratio,
        )
        input_size = in_channels * resolution ** 2
        representation_size = cfg.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM
        use_gn = cfg.MODEL.ROI_BOX_HEAD.USE_GN
        self.pooler = pooler
        self.fc6 = make_fc(input_size, representation_size, use_gn)
        self.fc7 = make_fc(representation_size, representation_size, use_gn)
        self.out_channels = representation_size

    def forward(self, x, proposals):
        x = self.pooler(x, proposals)
        x = x.view(x.size(0), -1)

        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))

        return x


@registry.ROI_BOX_FEATURE_EXTRACTORS.register("FPNXconv1fcFeatureExtractor")
class FPNXconv1fcFeatureExtractor(nn.Module):
    """
    Heads for FPN for classification
    """

    def __init__(self, cfg, in_channels):
        super(FPNXconv1fcFeatureExtractor, self).__init__()

        resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        scales = cfg.MODEL.ROI_BOX_HEAD.POOLER_SCALES
        sampling_ratio = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler = Pooler(
            output_size=(resolution, resolution),
            scales=scales,
            sampling_ratio=sampling_ratio,
        )
        self.pooler = pooler

        use_gn = cfg.MODEL.ROI_BOX_HEAD.USE_GN
        conv_head_dim = cfg.MODEL.ROI_BOX_HEAD.CONV_HEAD_DIM
        num_stacked_convs = cfg.MODEL.ROI_BOX_HEAD.NUM_STACKED_CONVS
        dilation = cfg.MODEL.ROI_BOX_HEAD.DILATION

        xconvs = []
        for ix in range(num_stacked_convs):
            xconvs.append(
                nn.Conv2d(
                    in_channels,
                    conv_head_dim,
                    kernel_size=3,
                    stride=1,
                    padding=dilation,
                    dilation=dilation,
                    bias=False if use_gn else True
                )
            )
            in_channels = conv_head_dim
            if use_gn:
                xconvs.append(group_norm(in_channels))
            xconvs.append(nn.ReLU(inplace=True))

        self.add_module("xconvs", nn.Sequential(*xconvs))
        for modules in [self.xconvs,]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    if not use_gn:
                        torch.nn.init.constant_(l.bias, 0)

        input_size = conv_head_dim * resolution ** 2
        representation_size = cfg.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM
        self.fc6 = make_fc(input_size, representation_size, use_gn=False)
        self.out_channels = representation_size

    def forward(self, x, proposals):
        x = self.pooler(x, proposals)
        x = self.xconvs(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc6(x))
        return x


def make_roi_box_feature_extractor(cfg, in_channels):
    func = registry.ROI_BOX_FEATURE_EXTRACTORS[
        cfg.MODEL.ROI_BOX_HEAD.FEATURE_EXTRACTOR
    ]
    return func(cfg, in_channels)
