#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import torch
import torch.nn as nn

from .darknet import Darknet
from .network_blocks import BaseConv
from tensorflow.keras.layers import UpSampling2D,Concatenate

class YOLOFPN(nn.Module):
    """
    YOLOFPN module. Darknet 53 is the default backbone of this model.
    """

    def __init__(
        self,
        depth=53,
        in_features=["dark3", "dark4", "dark5"],
    ):
        super().__init__()

        self.backbone = Darknet(depth)
        self.in_features = in_features

        # out 1
        self.out1_cbl = self._make_cbl(512, 256, 1)
        self.out1_list=self._make_embedding([256, 512], 512 + 256)
        self.out1 =nn.Sequential(*self.out1_list)

        # out 2
        self.out2_cbl = self._make_cbl(256, 128, 1)
        self.out2_list=self._make_embedding([128, 256], 256 + 128)
        self.out2 =nn.Sequential(*self.out2_list)

        # upsample
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

    def _make_cbl(self, _in, _out, ks):
        return BaseConv(_in, _out, ks, stride=1, act="lrelu")

    def _make_embedding(self, filters_list, in_filters):
        m =[
                self._make_cbl(in_filters, filters_list[0], 1),
                self._make_cbl(filters_list[0], filters_list[1], 3),
                self._make_cbl(filters_list[1], filters_list[0], 1),
                self._make_cbl(filters_list[0], filters_list[1], 3),
                self._make_cbl(filters_list[1], filters_list[0], 1),
            ]
        return m

    def load_pretrained_model(self, filename="./weights/darknet53.mix.pth"):
        with open(filename, "rb") as f:
            state_dict = torch.load(f, map_location="cpu")
        print("loading pretrained weights...")
        self.backbone.load_state_dict(state_dict)

    def forward(self, inputs):
        """
        Args:
            inputs (Tensor): input image.

        Returns:
            Tuple[Tensor]: FPN output features..
        """
        #  backbone
        out_features = self.backbone(inputs)
        x2, x1, x0 = [out_features[f] for f in self.in_features]

        #  yolo branch 1
        x1_in = self.out1_cbl(x0)
        x1_in = self.upsample(x1_in)
        x1_in = torch.cat([x1_in, x1], 1)
        out_dark4 = self.out1(x1_in)

        #  yolo branch 2
        x2_in = self.out2_cbl(out_dark4)
        x2_in = self.upsample(x2_in)
        x2_in = torch.cat([x2_in, x2], 1)
        out_dark3 = self.out2(x2_in)

        outputs = (out_dark3, out_dark4, x0)
        return outputs

    def keras_tensor(self,inputs):
        out_features = self.backbone.keras_tensor(inputs)
        x2, x1, x0 = [out_features[f] for f in self.in_features]

        #  yolo branch 1
        x1_in = self.out1_cbl.keras_tensor(x0)
        x1_in = UpSampling2D()(x1_in)
        x1_in=Concatenate()([x1_in, x1])
        for out1 in self.out1_list:
            x1_in=out1.keras_tensor(x1_in)
        out_dark4 =x1_in

        #  yolo branch 2
        x2_in = self.out2_cbl.keras_tensor(out_dark4)
        x2_in = UpSampling2D()(x2_in)
        x2_in = Concatenate()([x2_in, x2])
        for out2 in self.out2_list:
            x2_in=out2.keras_tensor(x2_in)
        out_dark3 =x2_in

        outputs = (out_dark3, out_dark4, x0)
        return outputs

    def keras_update(self):
        self.backbone.keras_update()
        self.out1_cbl.keras_update()
        for out1 in self.out1_list:
            out1.keras_update()
        self.out2_cbl.keras_update()
        for out2 in self.out2_list:
            out2.keras_update()
