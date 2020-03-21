import time
from enum import Enum
from functools import reduce
import contextlib
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable

from second.pytorch.models.voxelnet import register_voxelnet, VoxelNet
from second.pytorch.models import rpn


class SmallObjectHead(nn.Module):
    def __init__(self, num_filters, num_class, num_anchor_per_loc,
                 box_code_size, num_direction_bins, use_direction_classifier,
                 encode_background_as_zeros):
        super().__init__()
        self._num_anchor_per_loc = num_anchor_per_loc
        self._num_direction_bins = num_direction_bins
        self._num_class = num_class
        self._use_direction_classifier = use_direction_classifier
        self._box_code_size = box_code_size
        if encode_background_as_zeros:
            num_cls = num_anchor_per_loc * num_class
        else:
            num_cls = num_anchor_per_loc * (num_class + 1)

        self.net = nn.Sequential(
            nn.Conv2d(num_filters, 64, 3, bias=False, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, bias=False, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, bias=False, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        final_num_filters = 64
        self.conv_cls = nn.Conv2d(final_num_filters, num_cls, 1)
        self.conv_box = nn.Conv2d(final_num_filters,
                                  num_anchor_per_loc * box_code_size, 1)
        if use_direction_classifier:
            self.conv_dir_cls = nn.Conv2d(
                final_num_filters, num_anchor_per_loc * num_direction_bins, 1)

    def forward(self, x):
        x = self.net(x)
        # print(f'small head input shape: {x.shape}')
        batch_size = x.shape[0]
        box_preds = self.conv_box(x)
        cls_preds = self.conv_cls(x)
        # [N, C, y(H), x(W)]
        C, H, W = box_preds.shape[1:]
        box_preds = box_preds.view(-1, self._num_anchor_per_loc,
                                   self._box_code_size, H, W).permute(
                                       0, 1, 3, 4, 2).contiguous()
        cls_preds = cls_preds.view(-1, self._num_anchor_per_loc,
                                   self._num_class, H, W).permute(
                                       0, 1, 3, 4, 2).contiguous()
        ret_dict = {
            "box_preds": box_preds.view(batch_size, -1, self._box_code_size),
            "cls_preds": cls_preds.view(batch_size, -1, self._num_class),
        }
        if self._use_direction_classifier:
            dir_cls_preds = self.conv_dir_cls(x)
            dir_cls_preds = dir_cls_preds.view(
                -1, self._num_anchor_per_loc, self._num_direction_bins, H,
                W).permute(0, 1, 3, 4, 2).contiguous()
            ret_dict["dir_cls_preds"] = dir_cls_preds.view(batch_size, -1, self._num_direction_bins)
        return ret_dict


class DefaultHead(nn.Module):
    def __init__(self, num_filters, num_class, num_anchor_per_loc,
                 box_code_size, num_direction_bins, use_direction_classifier,
                 encode_background_as_zeros):
        super().__init__()
        self._num_anchor_per_loc = num_anchor_per_loc
        self._num_direction_bins = num_direction_bins
        self._num_class = num_class
        self._use_direction_classifier = use_direction_classifier
        self._box_code_size = box_code_size
        if encode_background_as_zeros:
            num_cls = num_anchor_per_loc * num_class
        else:
            num_cls = num_anchor_per_loc * (num_class + 1)

        final_num_filters = num_filters
        self.conv_cls = nn.Conv2d(final_num_filters, num_cls, 1)
        self.conv_box = nn.Conv2d(final_num_filters,
                                  num_anchor_per_loc * box_code_size, 1)
        if use_direction_classifier:
            self.conv_dir_cls = nn.Conv2d(
                final_num_filters, num_anchor_per_loc * num_direction_bins, 1)

    def forward(self, x):
        # print(f'large(default) head input shape: {x.shape}')
        batch_size = x.shape[0]
        box_preds = self.conv_box(x)
        cls_preds = self.conv_cls(x)
        # [N, C, y(H), x(W)]
        C, H, W = box_preds.shape[1:]
        box_preds = box_preds.view(-1, self._num_anchor_per_loc,
                                   self._box_code_size, H, W).permute(
                                       0, 1, 3, 4, 2).contiguous()
        cls_preds = cls_preds.view(-1, self._num_anchor_per_loc,
                                   self._num_class, H, W).permute(
                                       0, 1, 3, 4, 2).contiguous()
        ret_dict = {
            "box_preds": box_preds.view(batch_size, -1, self._box_code_size),
            "cls_preds": cls_preds.view(batch_size, -1, self._num_class),
        }
        if self._use_direction_classifier:
            dir_cls_preds = self.conv_dir_cls(x)
            dir_cls_preds = dir_cls_preds.view(
                -1, self._num_anchor_per_loc, self._num_direction_bins, H,
                W).permute(0, 1, 3, 4, 2).contiguous()
            ret_dict["dir_cls_preds"] = dir_cls_preds.view(batch_size, -1, self._num_direction_bins)
        return ret_dict

@register_voxelnet
class VoxelNetNuscenesMultiHead(VoxelNet):
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        assert self._num_class == 10
        assert isinstance(self.rpn, rpn.RPNNoHead)
        self.small_classes = ["pedestrian", "traffic_cone", "bicycle", "motorcycle", "barrier"]
        self.large_classes = ["car", "truck", "trailer", "bus", "construction_vehicle"]
        for c in self.small_classes:
            print(f'num_anchors_per_location_class({c}): {self.target_assigner.num_anchors_per_location_class(c)}]')
        for c in self.large_classes:
            print(f'num_anchors_per_location_class({c}): {self.target_assigner.num_anchors_per_location_class(c)}]')
        small_num_anchor_loc = sum([self.target_assigner.num_anchors_per_location_class(c) for c in self.small_classes])
        large_num_anchor_loc = sum([self.target_assigner.num_anchors_per_location_class(c) for c in self.large_classes])
        print(f'small_num_anchor_loc: {small_num_anchor_loc}')
        print(f'large_num_anchor_loc: {large_num_anchor_loc}')
        print(f'self._box_coder.code_size: {self._box_coder.code_size}')
        self.small_head = SmallObjectHead(
            num_filters=self.rpn._num_filters[0],
            num_class=self._num_class,
            num_anchor_per_loc=small_num_anchor_loc,
            encode_background_as_zeros=self._encode_background_as_zeros,
            use_direction_classifier=self._use_direction_classifier,
            box_code_size=self._box_coder.code_size,
            num_direction_bins=self._num_direction_bins,
        )
        self.large_head = DefaultHead(
            num_filters=np.sum(self.rpn._num_upsample_filters),
            num_class=self._num_class,
            num_anchor_per_loc=large_num_anchor_loc,
            encode_background_as_zeros=self._encode_background_as_zeros,
            use_direction_classifier=self._use_direction_classifier,
            box_code_size=self._box_coder.code_size,
            num_direction_bins=self._num_direction_bins,
        )

    def network_forward(self, voxels, num_points, coors, batch_size):
        self.start_timer("voxel_feature_extractor")
        voxel_features = self.voxel_feature_extractor(voxels, num_points,
                                                      coors)
        self.end_timer("voxel_feature_extractor")

        self.start_timer("middle forward")
        spatial_features = self.middle_feature_extractor(
            voxel_features, coors, batch_size)
        self.end_timer("middle forward")
        self.start_timer("rpn forward")
        rpn_out = self.rpn(spatial_features)
        # for k in rpn_out:
        #     print(f'rpn_out[{k}].shape: {rpn_out[k].shape}')
        r1 = rpn_out["stage0"]
        _, _, H, W = r1.shape
        # print(f'r1.shape: {r1.shape}')
        # Peiyun: this is basically trying to cut a 80% window
        cropsize40x40 = np.round(H * 0.1).astype(np.int64)
        # print(f'cropsize40x40: {cropsize40x40}')
        r1 = r1[:, :, cropsize40x40:-cropsize40x40, cropsize40x40:-cropsize40x40]
        small = self.small_head(r1)
        large = self.large_head(rpn_out["out"])
        self.end_timer("rpn forward")
        # concated preds MUST match order in class_settings in config.
        res = {
            "box_preds": torch.cat([large["box_preds"], small["box_preds"]], dim=1),
            "cls_preds": torch.cat([large["cls_preds"], small["cls_preds"]], dim=1),
        }
        if self._use_direction_classifier:
            res["dir_cls_preds"] = torch.cat([large["dir_cls_preds"], small["dir_cls_preds"]], dim=1)
        # print(f'large["box_preds"].shape: {large["box_preds"].shape}')
        # print(f'small["box_preds"].shape: {small["box_preds"].shape}')
        # print(f'large["cls_preds"].shape: {large["cls_preds"].shape}')
        # print(f'small["cls_preds"].shape: {small["cls_preds"].shape}')
        # print(f'large["dir_cls_preds"].shape: {large["dir_cls_preds"].shape}')
        # print(f'small["dir_cls_preds"].shape: {small["dir_cls_preds"].shape}')
        return res


class VisibilityFeatureExtractor(nn.Module):
    # TODO: HOOK UP MAX_SWEEPS CONFIG
    def __init__(self,
                 voxel_size=(0.25, 0.25, 8),
                 pc_range=(-50,-50,-5,50,50,3),
                #  num_filters=64,
                 max_sweeps=10):
        super().__init__()
        self.name = 'VisibilityFeatureNet'

        self.voxel_size = np.min(voxel_size)
        self.pc_range = pc_range

        px_min, py_min, pz_min, px_max, py_max, pz_max = pc_range
        self.x_size = int(np.ceil((px_max - px_min) / self.voxel_size))
        self.y_size = int(np.ceil((py_max - py_min) / self.voxel_size))
        self.z_size = int(np.ceil((pz_max - pz_min) / self.voxel_size))
        self.max_sweeps = max_sweeps
        
        #
        # self.num_filters = num_filters

        #
        # self._OCCUPIED = 1
        # self._UNKNOWN = 0 
        # self._FREE = -1

        # initialize with octomap's parameters
        # T = self.max_sweeps+1

        # we assume logodds for different timestamps can be different
        # and we learn what the optimal ones are
        # self.logodds_occupied = torch.nn.Parameter(torch.Tensor(1, T, 1))
        # self.logodds_occupied.data.fill_(np.log(0.7/(1-0.7)))  # probability = 0.7, approximately 0.85
        # self.logodds_free = torch.nn.Parameter(torch.Tensor(1, T, 1))
        # self.logodds_free.data.fill_(np.log(0.4/(1-0.4))) 
        # self.logodds_unknown = torch.nn.Parameter(torch.Tensor([0.0]), requires_grad=False) # probability = 0.4, approxiamtely -0.4
        
        # # setup a 1x1 filter along z dimension to mix visibilities across height
        # self.filter = torch.nn.Conv2d(in_channels=self.z_size, 
        #                               out_channels=self.num_filters, 
        #                               kernel_size=1, 
        #                               stride=1, 
        #                               padding=0, 
        #                               dilation=1, 
        #                               groups=1,
        #                               bias=True)

    def forward(self, logodds):
        # visibility: N x 2 x T x M (T[max]=11)
        # T = visibility.shape[1]
        
        # sum logodds across time with learnable parameters
        # logodds = torch.where(
        #     visibility == self._OCCUPIED, self.logodds_occupied[:T], torch.where(
        #         visibility == self._FREE, self.logodds_free[:T], self.logodds_unknown
        #     )
        # ).sum(axis=1).reshape((-1, self.z_size, self.y_size, self.x_size))
        
        # turn logodds into occupancy probabilities
        occupancy = torch.sigmoid(logodds)
        
        occupancy = occupancy.reshape((-1, self.z_size, self.y_size, self.x_size))

        return occupancy

        # filter occupancy probabilities
        # features = self.filter(occupancy)

        # return features


@register_voxelnet
class VoxelNetNuscenesMultiHeadVPN(VoxelNet):
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        assert self._num_class == 10
        assert isinstance(self.rpn, rpn.RPNNoHead)
        self.small_classes = ["pedestrian", "traffic_cone", "bicycle", "motorcycle", "barrier"]
        self.large_classes = ["car", "truck", "trailer", "bus", "construction_vehicle"]

        small_num_anchor_loc = sum([self.target_assigner.num_anchors_per_location_class(c) for c in self.small_classes])
        large_num_anchor_loc = sum([self.target_assigner.num_anchors_per_location_class(c) for c in self.large_classes])
        
        num_filters_small_head = self.rpn._num_filters[0]
        self.small_head = SmallObjectHead(
            num_filters=num_filters_small_head,
            num_class=self._num_class,
            num_anchor_per_loc=small_num_anchor_loc,
            encode_background_as_zeros=self._encode_background_as_zeros,
            use_direction_classifier=self._use_direction_classifier,
            box_code_size=self._box_coder.code_size,
            num_direction_bins=self._num_direction_bins,
        )

        num_filters_large_head = np.sum(self.rpn._num_upsample_filters)
        self.large_head = DefaultHead(
            num_filters=num_filters_large_head,
            num_class=self._num_class,
            num_anchor_per_loc=large_num_anchor_loc,
            encode_background_as_zeros=self._encode_background_as_zeros,
            use_direction_classifier=self._use_direction_classifier,
            box_code_size=self._box_coder.code_size,
            num_direction_bins=self._num_direction_bins,
        )

        # since the output goes straight into VPN 
        # we use the input dimension of VPN to set the output dimension 
        self.visibility_feature_extractor = VisibilityFeatureExtractor(
            voxel_size=self.voxel_generator.voxel_size,
            pc_range=self.voxel_generator.point_cloud_range, 
            # num_filters=self.vpn._num_input_features
        )

    def network_forward(self, voxels, num_points, coors, batch_size, logodds):
        self.start_timer("voxel_feature_extractor")
        voxel_features = self.voxel_feature_extractor(voxels, num_points, coors)
        self.end_timer("voxel_feature_extractor")
        # print(f"voxel features shape: {voxel_features.shape}")

        self.start_timer("middle forward")
        point_features = self.middle_feature_extractor(voxel_features, coors, batch_size)
        self.end_timer("middle forward")
        # print(f"spatial features shape: {point_features.shape}")

        self.start_timer("vfn forward")
        visibility_features = self.visibility_feature_extractor(logodds)
        self.end_timer("vfn forward")
        # print(f"visibility features shape: {visibility_features.shape}")

        self.start_timer("rpn forward")
        fused_features = torch.cat((point_features, visibility_features), dim=1)
        # rpn_out = self.rpn(point_features)
        # vpn_out = self.vpn(visibility_features)
        rpn_out = self.rpn(fused_features)

        r1 = rpn_out["stage0"]
        # v1 = vpn_out["stage0"]
        _, _, rH, rW = r1.shape
        # _, _, vH, vW = v1.shape

        # Peiyun: we run small head over 80% of the region at 2x resolution
        rcropsize40x40 = np.round(rH * 0.1).astype(np.int64)
        # vcropsize40x40 = np.round(vH * 0.1).astype(np.int64)
        # assert(rcropsize40x40 == vcropsize40x40)

        r1 = r1[:, :, rcropsize40x40:-rcropsize40x40, rcropsize40x40:-rcropsize40x40]
        # v1 = v1[:, :, vcropsize40x40:-vcropsize40x40, vcropsize40x40:-vcropsize40x40]

        # r1.shape: torch.Size([3, 64, 160, 160])
        # v1.shape: torch.Size([3, 32, 160, 160])
        # rpn_out["out"].shape: torch.Size([3, 384, 100, 100])
        # vpn_out["out"].shape: torch.Size([3, 192, 100, 100])
        # small = self.small_head(torch.cat([r1, v1], dim=1))
        # large = self.large_head(torch.cat([rpn_out["out"], vpn_out["out"]], dim=1))
        small = self.small_head(r1)
        large = self.large_head(rpn_out["out"])
        self.end_timer("rpn forward")

        # concated preds MUST match order in class_settings in config.
        res = {
            "box_preds": torch.cat([large["box_preds"], small["box_preds"]], dim=1),
            "cls_preds": torch.cat([large["cls_preds"], small["cls_preds"]], dim=1),
        }
        if self._use_direction_classifier:
            res["dir_cls_preds"] = torch.cat([large["dir_cls_preds"], small["dir_cls_preds"]], dim=1)
        return res
