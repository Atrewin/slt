#!/usr/bin/env python
# @jinhui
def add_project_root():
    import sys
    from os.path import abspath, join, dirname
    sys.path.insert(0, abspath(join(abspath(dirname(__file__)), '../')))
add_project_root()

import torch
torch.backends.cudnn.deterministic = True

import argparse
import numpy as np
import os
import shutil
import time
import queue

from signjoey.model import build_model
from signjoey.batch import Batch
from signjoey.helpers import (
    log_data_info,
    load_config,
    log_cfg,
    load_checkpoint,
    make_model_dir,
    make_logger,
    set_seed,
    symlink_update,
    mkdir,
)
from signjoey.model import SignModel
from signjoey.prediction import validate_on_data
from signjoey.loss import XentLoss
from signjoey.data import load_data, make_data_iter
from signjoey.builders import build_optimizer, build_scheduler, build_gradient_clipper
from signjoey.prediction import test, testing_exp
from signjoey.metrics import wer_single
from signjoey.vocabulary import SIL_TOKEN
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from torchtext.data import Dataset
from typing import List, Dict



import random
class MP_action():
    def __init__(self):
        pass


    def random_neighbor_add_frames(self, data):
        examples = data.examples

        for example_index in range(len(examples)):
            example = examples[example_index]
            sgn_fearture = example.sgn #list
            raw_len = len(sgn_fearture)
            add_num = int(raw_len * 0.10)

            start_index = random.randint(0, raw_len - add_num)

            end_index = start_index + add_num
            expand_sgn_fearture = []
            for index in range(raw_len):
                expand_sgn_fearture.append(sgn_fearture[index])
                if start_index <= index and index < end_index:
                    expand_sgn_fearture.append(sgn_fearture[index])
            example.sgn = expand_sgn_fearture
        return data


    def random_neighbor_decrease_frames(self, data):
        examples = data.examples

        for example_index in range(len(examples)):
            example = examples[example_index]
            sgn_fearture = example.sgn #list
            raw_len = len(sgn_fearture)
            add_num = int(raw_len * 5)

            start_index = random.randint(0, raw_len - add_num)

            end_index = start_index + add_num
            expand_sgn_fearture = []
            for index in range(raw_len):

                if start_index <= index and index < end_index:
                    if index % 2 == 0:
                        continue
                expand_sgn_fearture.append(sgn_fearture[index])
            example.sgn = expand_sgn_fearture
        return data

    def neighbor_add_frames(self, data, expend_times=2):
        examples = data.examples

        for example_index in range(len(examples)):
            example = examples[example_index]
            sgn_fearture = example.sgn #tensor
            expand_sgn_fearture = [0]*(len(sgn_fearture) * expend_times)

            for sgn_fearture_index in range(len(sgn_fearture)):
                for time in range(expend_times):
                    expand_sgn_fearture_index = sgn_fearture_index*expend_times + time
                    expand_sgn_fearture[expand_sgn_fearture_index] = sgn_fearture[sgn_fearture_index]
            # 这里利用到了指针的效果
            example.sgn = expand_sgn_fearture

        return data

    def random_add_frames(self, data):
        examples = data.examples

        for example_index in range(len(examples)):
            example = examples[example_index]
            sgn_fearture = example.sgn #list
            raw_len = len(sgn_fearture)
            add_num = int(raw_len * 0.10)

            repected_index = list(range(0, raw_len))
            random.shuffle(repected_index)

            repected_index = repected_index[0:add_num]
            expand_sgn_fearture = []
            for index in range(raw_len):
                expand_sgn_fearture.append(sgn_fearture[index])
                if index in repected_index:
                    expand_sgn_fearture.append(sgn_fearture[index])
            example.sgn = expand_sgn_fearture
        return data

    def random_decrease_frames(self, data):
        examples = data.examples

        for example_index in range(len(examples)):
            example = examples[example_index]
            sgn_fearture = example.sgn  # list
            raw_len = len(sgn_fearture)
            add_num = int(raw_len * 0.1)

            delete_index = list(range(0, raw_len))
            random.shuffle(delete_index)
            delete_index = delete_index[0:add_num]
            expand_sgn_fearture = []
            for index in range(raw_len):
                if index in delete_index:
                    continue
                expand_sgn_fearture.append(sgn_fearture[index])
            example.sgn = expand_sgn_fearture
        return data

    def neighbor_decrease_frames(self, data, skip_step=2):
        examples = data.examples

        for example_index in range(len(examples)):
            example = examples[example_index]
            sgn_fearture = example.sgn  # tensor
            decrease_sgn_fearture = []

            for sgn_fearture_index in range(0,len(sgn_fearture), skip_step):
                decrease_sgn_fearture.append(sgn_fearture[sgn_fearture_index])
            # 这里利用到了指针的效果
            example.sgn = decrease_sgn_fearture

        return data


    def ramdom_repeated_interval(self, data, repeated_interval = 4):

        examples = data.examples


        for example_index in range(len(examples)):
            example = examples[example_index]
            sgn_fearture = example.sgn  # tensor
            repeated_sgn_fearture = []
            repeated_start_index = random.randint(0, len(sgn_fearture)-5)

            for sgn_fearture_index in range(0, len(sgn_fearture)):
                if sgn_fearture_index == repeated_start_index:
                    for i in range(repeated_start_index, repeated_start_index+repeated_interval):
                        repeated_sgn_fearture.append(sgn_fearture[i])

                repeated_sgn_fearture.append(sgn_fearture[sgn_fearture_index])
            # 这里利用到了指针的效果
            example.sgn = repeated_sgn_fearture

        return data

    def ramdom_cut_interval(self, data, cut_interval = 4):

        examples = data.examples


        for example_index in range(len(examples)):
            example = examples[example_index]
            sgn_fearture = example.sgn  # tensor
            repeated_sgn_fearture = []
            cut_start_index = random.randint(0, len(sgn_fearture)-5)

            for sgn_fearture_index in range(0, len(sgn_fearture)):
                if cut_start_index <= sgn_fearture_index and sgn_fearture_index < cut_start_index + cut_interval :
                    continue
                repeated_sgn_fearture.append(sgn_fearture[sgn_fearture_index])
            # 这里利用到了指针的效果
            example.sgn = repeated_sgn_fearture

        return data

    def change_dataset(self, raw_data, mp_type=None):
        if mp_type == "neighbor_add_frames":
            raw_data = self.neighbor_add_frames(data=raw_data)
        if mp_type == "random_add_frames":
            raw_data = self.random_add_frames(data=raw_data)
        if mp_type == "neighbor_decrease_frames":
            raw_data = self.neighbor_decrease_frames(data=raw_data)
        if mp_type == "ramdom_repeated_interval":
            raw_data = self.ramdom_repeated_interval(data=raw_data)
        if mp_type == "ramdom_cut_interval":
            raw_data = self.ramdom_cut_interval(data=raw_data)

        if mp_type == "random_decrease_frames":
            raw_data = self.random_decrease_frames(data=raw_data)


        return raw_data



# pylint: disable=too-many-instance-attributes



def test_jinhui(args=None):
    cfg_file = args.config

    # predict with the best model on validation and test
    # (if test data is available)
    ckpt = "/apdcephfs/share_916081/jinhuiye/Projects/slt/training_task/sign_model_seed_7/best.ckpt"
    # output_name = "best.IT_{:08d}".format(1)
    output_path = "/apdcephfs/share_916081/jinhuiye/Projects/slt/training_task/sign_model_seed_7/output/{}/".format(args.mp)
    mkdir(output_path)
    logger = make_logger(model_dir=output_path)
    # @jinhui
    testing_exp(cfg_file, ckpt=ckpt, output_path=output_path, logger=logger, args=args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Joey-NMT")
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        type=str,
        help="Training configuration file (yaml).",
    )

    parser.add_argument(
        "--mp",
        default="neighbor_add_frames",
        type=str,
        help="how to do mp on images features.",
    )
    parser.add_argument(
        "--gpu_id", type=str, default="0", help="gpu to run your job on"
    )
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    test_jinhui(args=args)
