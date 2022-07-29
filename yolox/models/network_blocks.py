#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import torch
import torch.nn as nn
from tensorflow.keras.layers import Activation,LeakyReLU,Conv2D,DepthwiseConv2D,BatchNormalization,Add,Concatenate,UpSampling2D,ZeroPadding2D,Multiply,MaxPooling2D
import numpy as np

class SiLU(nn.Module):
    """export-friendly version of nn.SiLU()"""

    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)


def get_activation(name="silu", inplace=True):
    if name == "silu":
        module = nn.SiLU(inplace=inplace)
    elif name == "relu":
        module = nn.ReLU(inplace=inplace)
    elif name == "lrelu":
        module = nn.LeakyReLU(0.1015625, inplace=inplace)
    else:
        raise AttributeError("Unsupported act type: {}".format(name))
    return module


class BaseConv(nn.Module):
    """A Conv2d -> Batchnorm -> silu/leaky relu block"""

    def __init__(
        self, in_channels, out_channels, ksize, stride, groups=1, bias=False, act="silu"
    ):
        super().__init__()
        # same padding
        self.pad = (ksize - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=ksize,
            stride=stride,
            padding=self.pad,
            groups=groups,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = get_activation(act, inplace=True)

        self.keras_pad=ZeroPadding2D(padding=(self.pad,self.pad))
        self.groups=groups

        if groups==1:
            self.keras_conv=Conv2D(filters=out_channels,kernel_size=ksize,strides=stride,padding='valid',use_bias=bias)
        else:
            self.keras_conv=DepthwiseConv2D(kernel_size=ksize,strides=stride,padding="valid",use_bias=bias)
        self.keras_bn=BatchNormalization()
        self.act_name=act
        if act=='silu':
            self.keras_act=Activation(activation="sigmoid")
        elif act=="relu":
            self.keras_act=Activation(activation="relu")
        elif act=="lrelu":
            self.keras_act=LeakyReLU(0.1015625)
        else:
            self.keras_act=Activation(activation="linear")

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))

    def keras_tensor(self,x):
        if self.pad!=0:
            x=self.keras_pad(x)
        x=self.keras_conv(x)
        x=self.keras_bn(x)
        if self.act_name=="silu":
            y=self.keras_act(x)
            x=Multiply()([x,y])
        else:
            x=self.keras_act(x)
        return x

    def keras_update(self):
        cv=self.conv.state_dict()
        cv_w=cv['weight'].cpu().numpy().astype(np.float32)
        if self.groups==1:
            cv_w=np.transpose(cv_w,[2,3,1,0])
        else:
            cv_w=np.transpose(cv_w,[2,3,0,1])
        self.keras_conv.set_weights([cv_w])
        bn=self.bn.state_dict()
        bn_w=bn['weight'].cpu().numpy().astype(np.float32)
        bn_b=bn['bias'].cpu().numpy().astype(np.float32)
        bn_rm=bn['running_mean'].cpu().numpy().astype(np.float32)
        bn_rv=bn['running_var'].cpu().numpy().astype(np.float32)
        self.keras_bn.set_weights([bn_w,bn_b,bn_rm,bn_rv])

class DWConv(nn.Module):
    """Depthwise Conv + Conv"""

    def __init__(self, in_channels, out_channels, ksize, stride=1, act="silu"):
        super().__init__()
        self.dconv = BaseConv(
            in_channels,
            in_channels,
            ksize=ksize,
            stride=stride,
            groups=in_channels,
            act=act,
        )
        self.pconv = BaseConv(
            in_channels, out_channels, ksize=1, stride=1, groups=1, act=act
        )

    def forward(self, x):
        x = self.dconv(x)
        return self.pconv(x)
    
    def keras_tensor(self,x):
        x=self.dconv.keras_tensor(x)
        x=self.pconv.keras_tensor(x)
        return x

    def keras_update(self):
        self.dconv.keras_update()
        self.pconv.keras_update()
    
class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(
        self,
        in_channels,
        out_channels,
        shortcut=True,
        expansion=0.5,
        depthwise=False,
        act="silu",
    ):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        Conv = DWConv if depthwise else BaseConv
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv2 = Conv(hidden_channels, out_channels, 3, stride=1, act=act)
        self.use_add = shortcut and in_channels == out_channels

    def forward(self, x):
        y = self.conv2(self.conv1(x))
        if self.use_add:
            y = y + x
        return y
    def keras_tensor(self,x):
        y=self.conv1.keras_tensor(x)
        y=self.conv2.keras_tensor(y)
        if self.use_add:
            y=Add()([y,x])
        return y

    def keras_update(self):
        self.conv1.keras_update()
        self.conv2.keras_update()

class ResLayer(nn.Module):
    "Residual layer with `in_channels` inputs."

    def __init__(self, in_channels: int):
        super().__init__()
        mid_channels = in_channels // 2
        self.layer1 = BaseConv(
            in_channels, mid_channels, ksize=1, stride=1, act="lrelu"
        )
        self.layer2 = BaseConv(
            mid_channels, in_channels, ksize=3, stride=1, act="lrelu"
        )

    def forward(self, x):
        out = self.layer2(self.layer1(x))
        return x + out

    def keras_tensor(self,x):
        out=self.layer1.keras_tensor(x)
        out=self.layer2.keras_tensor(out)
        out=Add()([x,out])
        return out
    def keras_update(self):
        self.layer1.keras_update()
        self.layer2.keras_update()

class SPPBottleneck(nn.Module):
    """Spatial pyramid pooling layer used in YOLOv3-SPP"""

    def __init__(
        self, in_channels, out_channels, kernel_sizes=(3, 5, 7), activation="silu"
    ):
        super().__init__()
        hidden_channels = in_channels // 2
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=activation)
        self.m = nn.ModuleList(
            [
                nn.MaxPool2d(kernel_size=ks, stride=1, padding=ks // 2)
                for ks in kernel_sizes
            ]
        )
        conv2_channels = hidden_channels * (len(kernel_sizes) + 1)
        self.conv2 = BaseConv(conv2_channels, out_channels, 1, stride=1, act=activation)

        self.ksizes=kernel_sizes

    def forward(self, x):
        x = self.conv1(x)
        x = torch.cat([x] + [m(x) for m in self.m], dim=1)
        x = self.conv2(x)
        return x

    def keras_tensor(self,x):
        x=self.conv1.keras_tensor(x)
        y=[x]
        for ks in self.ksizes:
            pad=ks//2
            m=x
            if pad!=0:
                m=ZeroPadding2D(padding=(pad,pad))(x)
            m=MaxPooling2D(pool_size=(ks,ks),strides=(1,1))(m)
            y.append(m)
        x=Concatenate()(y)
        x=self.conv2.keras_tensor(x)
        return x

    def keras_update(self):
        self.conv1.keras_update()
        self.conv2.keras_update()

class CSPLayer(nn.Module):
    """C3 in yolov5, CSP Bottleneck with 3 convolutions"""

    def __init__(
        self,
        in_channels,
        out_channels,
        n=1,
        shortcut=True,
        expansion=0.5,
        depthwise=False,
        act="silu",
    ):
        """
        Args:
            in_channels (int): input channels.
            out_channels (int): output channels.
            n (int): number of Bottlenecks. Default value: 1.
        """
        # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        hidden_channels = int(out_channels * expansion)  # hidden channels
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv2 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv3 = BaseConv(2 * hidden_channels, out_channels, 1, stride=1, act=act)
        self.module_list = [
            Bottleneck(
                hidden_channels, hidden_channels, shortcut, 1.0, depthwise, act=act
            )
            for _ in range(n)
        ]
        self.m = nn.Sequential(*self.module_list)


    def forward(self, x):
        x_1 = self.conv1(x)
        x_2 = self.conv2(x)
        x_1 = self.m(x_1)
        x = torch.cat((x_1, x_2), dim=1)
        return self.conv3(x)

    def keras_tensor(self,x):
        x_1=self.conv1.keras_tensor(x)
        x_2=self.conv2.keras_tensor(x)
        for m in self.module_list:
            x_1=m.keras_tensor(x_1)
        x=Concatenate()([x_1,x_2])
        x=self.conv3.keras_tensor(x)
        return x

    def keras_update(self):
        self.conv1.keras_update()
        self.conv2.keras_update()
        for m in self.module_list:
            m.keras_update()
        self.conv3.keras_update()

class Focus(nn.Module):
    """Focus width and height information into channel space."""

    def __init__(self, in_channels, out_channels, ksize=1, stride=1, act="silu"):
        super().__init__()
        self.conv = BaseConv(in_channels * 4, out_channels, ksize, stride, act=act)
        self.in_channels=in_channels

    def forward(self, x):
        # shape of x (b,c,w,h) -> y(b,4c,w/2,h/2)
        patch_top_left = x[..., ::2, ::2]
        patch_top_right = x[..., ::2, 1::2]
        patch_bot_left = x[..., 1::2, ::2]
        patch_bot_right = x[..., 1::2, 1::2]
        x = torch.cat(
            (
                patch_top_left,
                patch_bot_left,
                patch_top_right,
                patch_bot_right,
            ),
            dim=1,
        )
        return self.conv(x)

    def keras_tensor(self,x):
        w0=np.ones(shape=(1,1,self.in_channels,1),dtype="float32")
        w0=np.expand_dims(w0,axis=0)
        x=DepthwiseConv2D(kernel_size=(1,1),strides=(1,1),use_bias=False,weights=w0)(x)
        
        w1=np.zeros(shape=(2,2,self.in_channels,1),dtype="float32")
        w1[0,0,...]=1.0
        w1=np.expand_dims(w1,axis=0)
        y1=DepthwiseConv2D(kernel_size=(2,2),strides=(2,2),use_bias=False,weights=w1,name="focus_dw1")(x)

        w2=np.zeros(shape=(2,2,self.in_channels,1),dtype="float32")
        w2[0,1,...]=1.0
        w2=np.expand_dims(w2,axis=0)
        y2=DepthwiseConv2D(kernel_size=(2,2),strides=(2,2),use_bias=False,weights=w2,name="focus_dw2")(x)

        w3=np.zeros(shape=(2,2,self.in_channels,1),dtype="float32")
        w3[1,0,...]=1.0
        w3=np.expand_dims(w3,axis=0)
        y3=DepthwiseConv2D(kernel_size=(2,2),strides=(2,2),use_bias=False,weights=w3,name="focus_dw3")(x)

        w4=np.zeros(shape=(2,2,self.in_channels,1),dtype="float32")
        w4[1,1,...]=1.0
        w4=np.expand_dims(w4,axis=0)
        y4=DepthwiseConv2D(kernel_size=(2,2),strides=(2,2),use_bias=False,weights=w4,name="focus_dw4")(x)

        x=Concatenate()([y1,y2,y3,y4])
        x=self.conv.keras_tensor(x)
        return x
        
    def keras_update(self):
        self.conv.keras_update()