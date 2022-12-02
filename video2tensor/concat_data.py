import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
from PIL import Image
import numpy as np
from overrides import overrides
import os
import cv2
import pickle
from tqdm import tqdm


# Setup proxy, pre-trained models need to be downloaded.
# os.environ['http_proxy'] = 'http://proxy.cse.cuhk.edu.hk:8000'
# os.environ['HTTP_PROXY'] = 'http://proxy.cse.cuhk.edu.hk:8000'
# os.environ['https_proxy'] = 'http://proxy.cse.cuhk.edu.hk:8000'
# os.environ['HTTPS_PROXY'] = 'http://proxy.cse.cuhk.edu.hk:8000'

# Device
DEVICE = torch.device('cpu')


def main(data_path=None, tip_str="", output_path="./"):
    files = os.listdir(data_path)
    data_list = []
    # cat data list
    for files_name in tqdm(files):
        if tip_str in files_name:
            embedded_file = os.path.join(data_path, files_name)

            try:

                # with open(embedded_file, 'rb') as fr:
                #     embedded_list = pickle.load(fr)
                embedded_list = torch.load(embedded_file)
                if 1 < len(embedded_list):
                    data_list.extend(embedded_list)
            except:
                print(embedded_file)

    # deplicating
    key_dict = {}
    data_list_deplicated = []
    for data in data_list:
        key = data["name"]
        if key not in key_dict.keys() and embedded_list[0]["sign"].size()[-1] == 71680:
            data_list_deplicated.append(data)

    if len(data_list_deplicated) > 0:
        save_name = tip_str + "_Ph4TResetnet_" + ".pickle"
        save_path = os.path.join(output_path, save_name)
        with open(save_path, "wb") as list_file:
            pickle.dump(data_list_deplicated, list_file)









    pass

import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser("Joey-NMT")

    parser.add_argument(
        "--data_root_path",
        default="/apdcephfs/share_916081/shared_info/zhengshguo/jinhui/Projects/SLT/data/EfficentNet0",
        type=str,
        help="Training configuration file (yaml).",
    )


    parser.add_argument(
        "--output_path",
        default="/apdcephfs/share_916081/shared_info/zhengshguo/jinhui/Projects/FinalData/EfficentNet",
        type=str,
        help="Training configuration file (yaml).",
    )

    args = parser.parse_args()

    main(data_path=args.data_root_path, tip_str="train", output_path=args.output_path)






