#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

from torch import nn

from .network_blocks import BaseConv, CSPLayer, DWConv, Focus, ResLayer, SPPBottleneck


class Darknet(nn.Module):
    # number of blocks from dark2 to dark5.
    depth2blocks = {21: [1, 2, 2, 1], 53: [2, 8, 8, 4]}

    def __init__(
        self,
        depth,
        in_channels=3,
        stem_out_channels=32,
        out_features=("dark3", "dark4", "dark5"),
    ):
        """
        Args:
            depth (int): depth of darknet used in model, usually use [21, 53] for this param.
            in_channels (int): number of input channels, for example, use 3 for RGB image.
            stem_out_channels (int): number of output channels of darknet stem.
                It decides channels of darknet layer2 to layer5.
            out_features (Tuple[str]): desired output layer name.
        """
        super().__init__()
        assert out_features, "please provide output features of Darknet"
        self.out_features = out_features
        self.stem_list=[BaseConv(in_channels, stem_out_channels, ksize=3, stride=1, act="lrelu"),*self.make_group_layer(stem_out_channels, num_blocks=1, stride=2)]
        self.stem = nn.Sequential(*self.stem_list)
        in_channels = stem_out_channels * 2  # 64

        num_blocks = Darknet.depth2blocks[depth]
        # create darknet with `stem_out_channels` and `num_blocks` layers.
        # to make model structure more clear, we don't use `for` statement in python.
        self.dark2_list=[*self.make_group_layer(in_channels, num_blocks[0], stride=2)]
        self.dark2 = nn.Sequential(*self.dark2_list)
        in_channels *= 2  # 128
        self.dark3_list=[*self.make_group_layer(in_channels, num_blocks[1], stride=2)]
        self.dark3 = nn.Sequential(*self.dark3_list)
        in_channels *= 2  # 256
        self.dark4_list=[*self.make_group_layer(in_channels, num_blocks[2], stride=2)]
        self.dark4 = nn.Sequential(*self.dark4_list)
        in_channels *= 2  # 512
        self.dark5_list=[*self.make_group_layer(in_channels, num_blocks[3], stride=2),*self.make_spp_block([in_channels, in_channels * 2], in_channels * 2)]
        self.dark5 = nn.Sequential(*self.dark5_list)

    def make_group_layer(self, in_channels: int, num_blocks: int, stride: int = 1):
        "starts with conv layer then has `num_blocks` `ResLayer`"
        return [
            BaseConv(in_channels, in_channels * 2, ksize=3, stride=stride, act="lrelu"),
            *[(ResLayer(in_channels * 2)) for _ in range(num_blocks)],
        ]

    def make_spp_block(self, filters_list, in_filters):
        m = [
                BaseConv(in_filters, filters_list[0], 1, stride=1, act="lrelu"),
                BaseConv(filters_list[0], filters_list[1], 3, stride=1, act="lrelu"),
                SPPBottleneck(
                    in_channels=filters_list[1],
                    out_channels=filters_list[0],
                    activation="lrelu",
                ),
                BaseConv(filters_list[0], filters_list[1], 3, stride=1, act="lrelu"),
                BaseConv(filters_list[1], filters_list[0], 1, stride=1, act="lrelu"),
            ]
        return m

    def forward(self, x):
        outputs = {}
        x = self.stem(x)
        outputs["stem"] = x
        x = self.dark2(x)
        outputs["dark2"] = x
        x = self.dark3(x)
        outputs["dark3"] = x
        x = self.dark4(x)
        outputs["dark4"] = x
        x = self.dark5(x)
        outputs["dark5"] = x
        return {k: v for k, v in outputs.items() if k in self.out_features}

    def keras_tensor(self,x):
        outputs={}
        for stem in self.stem_list:
            x=stem.keras_tensor(x)
        outputs["stem"] = x
        for dark2 in self.dark2_list:
            x=dark2.keras_tensor(x)
        outputs["dark2"] = x
        for dark3 in self.dark3_list:
            x = dark3.keras_tensor(x)
        outputs["dark3"] = x
        for dark4 in self.dark4_list:
            x=dark4.keras_tensor(x)
        outputs["dark4"] = x
        for dark5 in self.dark5_list:
            x=dark5.keras_tensor(x)
        outputs["dark5"] = x
        return {k: v for k, v in outputs.items() if k in self.out_features}

    def keras_update(self):
        for stem in self.stem_list:
            stem.keras_update()
        for dark2 in self.dark2_list:
            dark2.keras_update()
        for dark3 in self.dark3_list:
            dark3.keras_update()
        for dark4 in self.dark4_list:
            dark4.keras_update()
        for dark5 in self.dark5_list:
            dark5.keras_update()

class CSPDarknet(nn.Module):
    def __init__(
        self,
        dep_mul,
        wid_mul,
        out_features=("dark3", "dark4", "dark5"),
        depthwise=False,
        act="silu",
    ):
        super().__init__()
        assert out_features, "please provide output features of Darknet"
        self.out_features = out_features
        Conv = DWConv if depthwise else BaseConv

        base_channels = int(wid_mul * 64)  # 64
        base_depth = max(round(dep_mul * 3), 1)  # 3

        # stem
        self.stem = Focus(3, base_channels, ksize=3, act=act)

        # dark2
        self.dark2_list=[Conv(base_channels, base_channels * 2, 3, 2, act=act),CSPLayer(base_channels * 2,base_channels * 2,n=base_depth,depthwise=depthwise,act=act)]
        self.dark2 = nn.Sequential(*self.dark2_list)

        # dark3
        self.dark3_list=[Conv(base_channels * 2, base_channels * 4, 3, 2, act=act),CSPLayer(base_channels * 4,base_channels * 4,n=base_depth * 3,depthwise=depthwise,act=act)]
        self.dark3 = nn.Sequential(*self.dark3_list)

        # dark4
        self.dark4_list=[Conv(base_channels * 4, base_channels * 8, 3, 2, act=act),CSPLayer(base_channels * 8,base_channels * 8,n=base_depth * 3,depthwise=depthwise,act=act)]
        self.dark4 = nn.Sequential(*self.dark4_list)

        # dark5
        self.dark5_list=[Conv(base_channels * 8, base_channels * 16, 3, 2, act=act),SPPBottleneck(base_channels * 16, base_channels * 16, activation=act),
                        CSPLayer(base_channels * 16,base_channels * 16,n=base_depth,shortcut=False,depthwise=depthwise,act=act)]
        self.dark5 = nn.Sequential(*self.dark5_list)

    def forward(self, x):
        outputs = {}
        x = self.stem(x)
        outputs["stem"] = x
        x = self.dark2(x)
        outputs["dark2"] = x
        x = self.dark3(x)
        outputs["dark3"] = x
        x = self.dark4(x)
        outputs["dark4"] = x
        x = self.dark5(x)
        outputs["dark5"] = x
        return {k: v for k, v in outputs.items() if k in self.out_features}

    def keras_tensor(self, x):
        outputs = {}
        x=self.stem.keras_tensor(x)
        outputs["stem"] = x
        for dark2 in self.dark2_list:
            x = dark2.keras_tensor(x)
        outputs["dark2"] = x
        for dark3 in self.dark3_list:
            x=dark3.keras_tensor(x)
        outputs["dark3"] = x
        for dark4 in self.dark4_list:
            x=dark4.keras_tensor(x)
        outputs["dark4"] = x
        for dark5 in self.dark5_list:
            x=dark5.keras_tensor(x)
        outputs["dark5"] = x
        return {k: v for k, v in outputs.items() if k in self.out_features}

    def keras_update(self):
        self.stem.keras_update()
        for dark2 in self.dark2_list:
            dark2.keras_update()
        for dark3 in self.dark3_list:
            dark3.keras_update()
        for dark4 in self.dark4_list:
            dark4.keras_update()
        for dark5 in self.dark5_list:
            dark5.keras_update()