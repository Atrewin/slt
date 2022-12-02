# coding: utf-8
"""
Data module
"""

import os

import numpy as np
import json
from PIL import Image
import torch
from torchvision import transforms

from efficientnet_pytorch import EfficientNet
import cv2

# global device
# global model
# global image_size
# import ssl


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_name = 'efficientnet-b0'
print(device)
model = EfficientNet.from_pretrained(model_name).to(device)
image_size = EfficientNet.get_image_size(model_name)
def sign2tensor(video_path):
    '''
    function: obtain feature file of one img by efficientnet

    '''
    imgs = []
    tfms = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    '''
    # because of imput of efficientnet is 4 dim, so unsqueeze
    for image_index in os.listdir(video_frames_path):
         img_path = video_frames_path + '/' + image_index
         img = cv2.imread(img_path)
         img = cv2.resize(img, (600,600))

         imgs.append(img)

    imgs = np.array(imgs)
    #imgs = tfms(imgs) # torch.size([total_frames,3,600,600])
    imgs = torch.Tensor(imgs)
    imgs = imgs.transpose(1,3)
    #print("video_tensor shape" , imgs.size())
    '''
    for image_index in os.listdir(video_path):
        img_path = video_path + '/' + image_index
        img = tfms(Image.open(img_path)).unsqueeze(0).to(device)  # efficientnet-b0 input: 4 dim
        feature_tensor = model.extract_features(img)
        # print(feature_tensor.size())
        feature_np = feature_tensor.cpu().detach().numpy()
        # print(feature_np)
        imgs.append(feature_np)

    # using the simplest way(average) to from video tensor from  many frame pictures tensor
    # later, we can add more information like optical tensor、facial tensor to form one better video tensor
    imgs = np.array(imgs)
    # try idea: video frame sentence, not to reduce
    # imgs = np.mean(imgs, 0)

    features = torch.Tensor(imgs).to(device)
    # print("video tensor is :")
    print(features.size())
    # print(features)

    return features


import pandas as pd
import pickle


def read_csv_file(file_path=""):
    df = pd.read_csv(file_path, encoding="utf-8")

    return df


def format_PhoneDataset(
        csv_file="/home/yejinhui/Dataset/PHOENIX-2014-T/annotations/manual/PHOENIX-2014-T.train.corpus.csv",
        video_root_path="/home/yejinhui/Dataset/PHOENIX-2014-T/features/fullFrame-210x260px/",
        split_name="train",
        output_path="/datas/",
        start_index=2220):
    """


    Returns: [{name: "dev/11August_2010_Wednesday_tagesschau-2",
               signer: Signer08,
               gloss: "DRUCK TIEF KOMMEN" ,
               text: "tiefer luftdruck bestimmt in den nachsten tagen unser wetter",
               sign: Tensor [length, 1024]
               }]

    """
    answers = []
    # csv_file = "/home/yejinhui/Dataset/PHOENIX-2014-T/annotations/manual/PHOENIX-2014-T.train.corpus.csv"
    # 获取基础信息
    annotations_csv = read_csv_file(csv_file)

    # title = annotations_csv[0]
    # video_root_path = "/home/yejinhui/Dataset/PHOENIX-2014-T/features/fullFrame-210x260px/"
    split_name = split_name
    index = 0
    # start_index = 2220
    print(start_index)
    for row in annotations_csv.values:
        if index < start_index:
            index += 1
            continue
        try:
            row_list = list(row)[0].split("|")
            name = os.path.join(split_name, row_list[0])
            signer = row_list[4]
            gloss = row_list[5]
            text = row_list[6]

            video_frames_path = os.path.join(video_root_path, name)

            sign = sign2tensor(video_frames_path)
            sign = sign.reshape([sign.size()[0], -1])

            sample_format = {
                "name": name,
                "signer": signer,
                "gloss": gloss,
                "text": text,
                "sign": sign
            }

            answers.append(sample_format)

            if start_index + 1000 < index:
                break

        except:

            # 封装格式
            save_name = split_name + str(start_index) + "_Ph4T_" + str(index) + ".pickle"
            save_path = os.path.join(output_path, save_name)
            torch.save(answers, save_path)

            del answers
            answers = []
            pass

        index += 1


        if index % 50 == 0:
            save_name = split_name + str(start_index) + "_Ph4T_" + str(index) + ".pickle"
            save_path = os.path.join(output_path, save_name)
            torch.save(answers, save_path)

            pass
            del answers
            answers = []

    pass

    # 获取 sign tensor

    # 封装格式
    save_name = split_name + str(start_index) + "_Ph4T_" + str(index) + ".pickle"
    save_path = os.path.join(output_path, save_name)
    torch.save(answers, save_path)

    return None


def preprocess(file_name):
    '''
    function:  obtain feature file
    parameter: file_name path of train file. The structure of file_name is as follows:
               -file_name
                 all videos 1-n represents index
                 --annotation file 1:
                 --annotation_file 2:
                 --annotation_file 3:
                 ...
                 --annotation_file n:(n represents the total number of videos)
                   10 sign language video frames file
                   ---en_video_frames file
                   ---zh_video_frames file
                   ---is_video_frames file
                   ---bg_video_frames: containing all frames information of sign language video
                     ----seq_id = video_index
                     ----sgn_frames_pictures
                     ----sgn_lang = sign_lang_tag


    '''
    for video_index in os.listdir(file_name):
        annotation_file = file_name + '/' + video_index
        for sign_lang_tag in os.listdir(annotation_file):
            video_frames_path = annotation_file + '/' + sign_lang_tag

            feature_tensor = sign2tensor(video_frames_path)


import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Joey-NMT")
    parser.add_argument(
        "--csv_file",
        default="/apdcephfs/share_916081/shared_info/zhengshguo/jinhui/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/annotations/manual/PHOENIX-2014-T.train.corpus.csv",
        type=str,
        help="Training configuration file (yaml).",
    )

    parser.add_argument(
        "--video_root_path",
        default="/apdcephfs/share_916081/shared_info/zhengshguo/jinhui/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/features/fullFrame-210x260px/",
        type=str,
        help="Training configuration file (yaml).",
    )

    parser.add_argument(
        "--split_name",
        default="train",
        type=str,
        help="Training configuration file (yaml).",
    )
    parser.add_argument(
        "--output_path",
        default="/apdcephfs/share_916081/shared_info/zhengshguo/jinhui/Projects/SLT/data/EfficentNet",
        type=str,
        help="Training configuration file (yaml).",
    )
    parser.add_argument(
        "--start_index", type=int, default=0, help="gpu to run your job on"
    )
    args = parser.parse_args()
    # test_file = '/home/yejinhui/Projects/SP-10/videos/dev_Preprocess/10004/de'
    # sign2tensor(test_file)

    csv_file = args.csv_file
    video_root_path = args.video_root_path
    output_path = args.output_path

    split_name = args.split_name
    format_PhoneDataset(csv_file=csv_file, video_root_path=video_root_path, split_name=split_name,
                        output_path=output_path, start_index=int(args.start_index))



'''

def load_dataset_file(filename):

class SignTranslationDataset(data.Dataset):
    """Defines a dataset for machine translation."""

    @staticmethod
    def sort_key(ex):
        return data.interleave_keys(len(ex.sgn), len(ex.txt))

    def __init__(
        self,
        path: str,
        fields: Tuple[RawField, RawField, Field, Field, Field, Field, Field],
        **kwargs
    ):
        """Create a SignTranslationDataset given paths and fields.

        Arguments:
            path: Common prefix of paths to the data files for both languages.
            exts: A tuple containing the extension to path for each language.
            fields: A tuple containing the fields that will be used for data
                in each language.
            Remaining keyword arguments: Passed to the constructor of
                data.Dataset.
        # data.py
        train_data = SignTranslationDataset(
        path=train_paths,  # ./data/all.train
        fields=(
            sequence_field,
            signer_field,
            lang_field,
            sgn_field,
            gls_field,
            txt_field,
            sign_lang_field,
        ),
        filter_pred=lambda x: len(vars(x)["sgn"]) <= max_sent_length
        and len(vars(x)["txt"]) <= max_sent_length,
           )

        """
        if not isinstance(fields[0], (tuple, list)):
            fields = [
                                                                                                                                                                                          21,0-1        10%
     """
        if not isinstance(fields[0], (tuple, list)):
            fields = [
                ("sequence", fields[0]),
                ("signer", fields[1]),
                ("lang", fields[2]),
                ("sgn", fields[3]),
                ("gls", fields[4]),
                ("txt", fields[5]),
                ("sign_lang", fields[6]),
            ]

        if not isinstance(path, list):
            path = [path]

        samples = {}
        # path: the file path. The file contains 10 sign languages frames file. eg zh_slt, de_slt...
        # annotation_file: contains many files. Each sample file contains sign language frames pic.
        for lang_tag in os.listdir(path): 
            #tmp = load_dataset_file(annotation_file)
            annotation_file = path + '/' + lang_tag # assumed run in linux os
            for s in os.list(annotation_file):
                seq_id = s["name"]
                if seq_id in samples:
                    assert samples[seq_id]["name"] == s["name"]
                    assert samples[seq_id]["signer"] == s["signer"]
                    assert samples[seq_id]["gloss"] == s["gloss"]
                    assert samples[seq_id]["text"] == s["text"]
                    assert samples[seq_id]["lang"] == s["lang"]
                    assert samples[seq_id]["sign_lang"] == s["sin_lang"]
                    samples[seq_id]["sign"] = torch.cat(
                        [samples[seq_id]["sign"], s["sign"]], axis=1
                    )
                else:
                    samples[seq_id] = {
                        "name": s["name"],
                        "signer": s["signer"],
                        "gloss": s["gloss"],
                        "text": s["text"],
                        "sign": s["sign"],
                        "lang": s["lang"],
                        "sign_lang": s["sin_lang"],
                    }
           '''
