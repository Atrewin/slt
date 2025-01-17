# coding: utf-8
"""
Data module
"""
from torchtext import data
# from torchtext.data import Field, RawField
from torchtext.data import Field, RawField
from typing import List, Tuple
import pickle
import gzip
import torch
import os
import numpy as np
import json
from PIL import Image
import torch
from torchvision import transforms

def load_dataset_file(filename):

    if "pickle" not in filename:
        with gzip.open(filename, "rb") as f:
            loaded_object = pickle.load(f)
    else:
        with open(filename, 'rb') as fr:
            loaded_object = pickle.load(fr)
    return loaded_object

class SignVideo2TextDataset(data.Dataset):
    """Defines a dataset for machine translation."""

    @staticmethod
    def sort_key(ex):
        return data.interleave_keys(len(ex.sgn), len(ex.txt))

    def __init__(
        self,
        path: str,
        fields: Tuple[RawField, RawField, Field, Field, Field],
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
        """
        if not isinstance(fields[0], (tuple, list)):
            fields = [
                ("sequence", fields[0]),
                ("signer", fields[1]),
                ("sgn", fields[2]),
                ("gls", fields[3]),
                ("txt", fields[4]),
            ]
        self.video_raw_path = "/apdcephfs/share_916081/shared_info/zhengshguo/jinhui/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/features/fullFrame-210x260px"
        if not isinstance(path, list):
            path = [path]

        samples = {}
        for annotation_file in path:
            tmp = load_dataset_file(annotation_file)
            for s in tmp:
                seq_id = s["name"]
                if seq_id in samples: #@jinhui 我觉得不应该合并数据 原作者探索过 feature cat
                    # assert samples[seq_id]["name"] == s["name"]
                    # assert samples[seq_id]["signer"] == s["signer"]
                    # assert samples[seq_id]["gloss"] == s["gloss"]
                    # assert samples[seq_id]["text"] == s["text"]
                    samples[seq_id]["sign"] = torch.cat(
                        [samples[seq_id]["sign"], s["sign"]], axis=1)

                    # seq_id = seq_id + "_Teacher"
                    # samples[seq_id] = {
                    #     "name": s["name"],
                    #     "signer": s["signer"],
                    #     "gloss": s["gloss"],
                    #     "text": s["text"],
                    #     "sign": s["sign"],
                    # }

                else:
                    samples[seq_id] = {
                        "name": s["name"],
                        "signer": s["signer"],
                        "gloss": s["gloss"],
                        "text": s["text"],
                        "sign": s["sign"],
                    }

        examples = []
        for s in samples:
            sample = samples[s]
            examples.append(
                data.Example.fromlist(
                    [
                        sample["name"],
                        sample["signer"],
                        # This is for numerical stability
                        sample["sign"] + 1e-8,
                        sample["gloss"].strip(),
                        sample["text"].strip(),
                    ],
                    fields,
                )
            )
        super().__init__(examples, fields, **kwargs)

        # raw_frames_imgs = []
        # for index, s in enumerate(samples):
        #     video_frame_file = os.path.join(self.video_raw_path, s)
        #
        #     imags = self.sign2tensor(video_path=video_frame_file)
        #     raw_frames_imgs.append(imags)
        #
        #     print(index)
        #
        # self.raw_frames_imgs = raw_frames_imgs



    # @overwrite
    def __getitem__(self, i):



        #@jinhui
        example = self.examples[i]

        # video_frame_file = os.path.join(self.video_raw_path, example.sequence)
        # #
        # imags = self.sign2tensor(video_path=video_frame_file)
        # #@jinhui waiting TODO
        # example.rawFrames = imags
        return example
    def sign2tensor(self, video_path):
        '''
        function: read a sign frames to tensor

        '''

        image_size = 1024
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
            img = tfms(Image.open(img_path)).unsqueeze(0) # efficientnet-b0 input: 4 dim

            imgs.append(img)

        # using the simplest way(average) to from video tensor from  many frame pictures tensor
        # later, we can add more information like optical tensor、facial tensor to form one better video tensor
        # imgs = np.array(imgs)
        # try idea: video frame sentence, not to reduce
        # features = torch.Tensor(imgs)

        return imgs

class SignTranslationDataset(data.Dataset):
    """Defines a dataset for machine translation."""

    @staticmethod
    def sort_key(ex):
        return data.interleave_keys(len(ex.sgn), len(ex.txt))

    def __init__(
        self,
        path: str,
        fields: Tuple[RawField, RawField, Field, Field, Field],
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
        """
        if not isinstance(fields[0], (tuple, list)):
            fields = [
                ("sequence", fields[0]),
                ("signer", fields[1]),
                ("sgn", fields[2]),
                ("gls", fields[3]),
                ("txt", fields[4]),
            ]

        if not isinstance(path, list):
            path = [path]

        samples = {}
        for annotation_file in path:
            tmp = load_dataset_file(annotation_file)
            for s in tmp:
                seq_id = s["name"]
                if seq_id in samples: #@jinhui 我觉得不应该合并数据 原作者探索过 multiview feature cat
                    # assert samples[seq_id]["name"] == s["name"]
                    # assert samples[seq_id]["signer"] == s["signer"]
                    # assert samples[seq_id]["gloss"] == s["gloss"]
                    # assert samples[seq_id]["text"] == s["text"]
                    # samples[seq_id]["sign"] = torch.cat(
                    #     [samples[seq_id]["sign"], s["sign"]], axis=1)
                    # print(seq_id)
                    seq_id = seq_id + "_Teacher"
                    samples[seq_id] = {
                        "name": s["name"],
                        "signer": s["signer"],
                        "gloss": s["gloss"],
                        "text": s["text"],
                        "sign": s["sign"],
                    }

                else:
                    samples[seq_id] = {
                        "name": s["name"],
                        "signer": s["signer"],
                        "gloss": s["gloss"],
                        "text": s["text"],
                        "sign": s["sign"],
                    }

        examples = []
        for s in samples:
            sample = samples[s]
            examples.append(
                data.Example.fromlist(
                    [
                        sample["name"],
                        sample["signer"],
                        # This is for numerical stability
                        sample["sign"] + 1e-8,
                        sample["gloss"].strip(),
                        sample["text"].strip(),
                    ],
                    fields,
                )
            )
        super().__init__(examples, fields, **kwargs)

    # @overwrite
    def __getitem__(self, i):
        #@jinhui 0425 我想要扩展 gloss
        # example = copy.deepcopy(self.examples[i])
        # ratio = max(len(example.sgn) // len(example.gls), 1)
        # example.gls = [elem for elem in example.gls for i in range(ratio)]
        # return example
        return self.examples[i]

#@jinhui
from signjoey.helpers import read_all_dataset
import copy
class Gloss2TextDataset(data.Dataset):
    """Defines a dataset for machine translation."""

    @staticmethod
    def sort_key(ex):
        return data.interleave_keys(len(ex.gls), len(ex.txt))

    def __init__(
        self,
        path: str,
        fields: Tuple[Field, Field],
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
        """
        if not isinstance(fields[0], (tuple, list)):
            fields = [
                ("gls", fields[0]),
                ("txt", fields[1]),
            ]

        if not isinstance(path, list):
            path = [path]

        samples = {}
        for annotation_file in path: #@jinhui wait
            train_g2t_path_list = annotation_file.split(":")
            tmp_src = read_all_dataset(train_g2t_path_list[0])
            tmp_tgt = read_all_dataset(train_g2t_path_list[1])
            for index, s in enumerate(tmp_src):
                seq_id = train_g2t_path_list[0] + "-" + str(index)
                # #@jinhui 数据对齐
                # if tmp_tgt[index][-1] != ".":
                #     tmp_tgt[index] = tmp_tgt[index][:-1] + " ."

                samples[seq_id] = {
                    "gloss": s,
                    "text": tmp_tgt[index],
                }

        examples = []
        for s in samples:
            sample = samples[s]
            examples.append(
                data.Example.fromlist(
                    [
                        sample["gloss"].strip(),
                        sample["text"].strip(),
                    ],
                    fields,
                )
            )
        super().__init__(examples, fields, **kwargs)

    # @overwrite
    def __getitem__(self, i):
        #@jinhui 0425 我想要扩展 gloss

        # example = copy.deepcopy(self.examples[i])
        #
        # # ratio = max(len(example.sgn) // len(example.gls), 1)
        # if len(example.gls) < 20:
        #     example.gls = [elem for elem in example.gls for i in range(6)]
        #
        # else:
        #     example = self.__getitem__((i+17) % len(self.examples))
        return self.examples[i]

if __name__ == "__main__":

    pass