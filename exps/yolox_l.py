#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import os

from yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 1.0
        self.width = 1.0
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        self.data_dir = "../../dataset"
        self.train_ann = "train.json"
        self.val_ann = "test.json"

        self.num_classes = 1
        self.input_size = (512, 512)
        self.test_size = (512, 512)
        self.max_epoch = 500
        self.data_num_workers = 4
        self.eval_interval = 1
        self.save_history_ckpt = False