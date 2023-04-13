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
import copy
from signjoey.dataset import load_dataset_file
from signjoey.helpers import read_all_dataset
def teacher_model_data_build(path=None):
    embedded_file= "/apdcephfs/share_916081/shared_info/zhengshguo/jinhui/Projects/SLT/data/PHOENIX2014T/phoenix14t.pami0.train_V_SMKD_1024.pickle"
    teacher_target_file = "/apdcephfs/share_916081/shared_info/zhengshguo/jinhui/Projects/SLT/data/PHOENIX2014T/ensemble_de.txt"
    target_file = "/apdcephfs/share_916081/shared_info/zhengshguo/jinhui/Projects/SLT/data/PHOENIX2014T/target.txt"
    teacher_target_file_list = [
        "/apdcephfs/share_916081/shared_info/zhengshguo/jinhui/Projects/SLT/data/PHOENIX2014T/EnsembleT1.txt",
        "/apdcephfs/share_916081/shared_info/zhengshguo/jinhui/Projects/SLT/data/PHOENIX2014T/EnsembleT12.txt",
        "/apdcephfs/share_916081/shared_info/zhengshguo/jinhui/Projects/SLT/data/PHOENIX2014T/EnsembleT34.txt"

    ]
    teacher_target_file_list.append(teacher_target_file)


    raw_datas = load_dataset_file(filename=embedded_file)
    target = read_all_dataset(filename=target_file)
    signs_len = len(raw_datas)
    for index in range(len(teacher_target_file_list)):
        teacher_target_file = teacher_target_file_list[index]
        teacher_target = read_all_dataset(filename=teacher_target_file)
        # assert signs_len == len(teacher_target) and len(teacher_target) == len(target)
        print(teacher_target_file)
        offset = 0
        for index in range(signs_len):
            output_text = raw_datas[index]["text"]
            target_text = target[index+offset]
            new_taget_text = teacher_target[index+offset]

            if output_text[0:-2] != target_text:
                offset += 1
                assert offset <= 1
                continue
                pass
            new_sample = copy.deepcopy(raw_datas[index])
            new_sample["text"] = new_taget_text + " ."
            new_sample["name"] = new_sample["name"] + "_teacher_{}".format(index)
            raw_datas.append(new_sample)


    save_path = embedded_file + "_Teachers_DA_V_SMKD" + ".pickle"
    with open(save_path, "wb") as list_file:
        pickle.dump(raw_datas, list_file)

def get_new_sign_featrue(path=None):
    embedded_file= "/home/yejinhui/Projects/SLT/data/PHOENIX2014T/phoenix14t.pami0.test"
    new_embeddin_file_root_path = "/apdcephfs/share_916081/shared_info/zhengshguo/jinhui/Projects/SLT/VAC_CSLR/checkpoints/2/"

    raw_datas = load_dataset_file(filename=embedded_file)
    signs_len = len(raw_datas)
    miss_examples = []
    for index in tqdm(list(range(signs_len))):

        example = raw_datas[index]

        name_id = example["name"]

        PH14T_file = name_id + "_features.npy"
        new_feature_file = os.path.join(new_embeddin_file_root_path, PH14T_file)
        if not os.path.exists(new_feature_file):
            miss_examples.append(new_feature_file)
            print(new_feature_file)
            continue

        new_embedded = np.load(new_feature_file, allow_pickle=True).item()
        new_embedded_features = new_embedded["features"]
        example["sign"] = new_embedded_features
        raw_datas[index-len(miss_examples)] = example

    print(len(miss_examples))
    save_path = embedded_file + "_V_SMKD_1024" + ".pickle"
    with open(save_path, "wb") as list_file:
        pickle.dump(raw_datas[:signs_len-len(miss_examples)], list_file)

import pandas
import numpy as np
def build_cslt_sign_featrue(path=None,dataset_type="train"):
    anno_path_2 = r"/home/yejinhui/Projects/SLT/VAC_CSLR/dataset/CSLT/sentence_label/video_map.txt"
    inputs_list_2 = pandas.read_csv(anno_path_2)
    inputs_list_split = pandas.read_csv("/home/yejinhui/Projects/SLT/VAC_CSLR/dataset/CSLT/sentence_label/split_1.txt")

    inputs_list_2 = (inputs_list_2.to_dict()[
                         "index|name|length|gloss|char|word|postag"].values())  # inputs_list = (inputs_list.to_dict()['name|video|start|end|speaker|orth|translation'].values())
    inputs_list_split = (inputs_list_split.to_dict()["name|split"].values())
    split_dict = {}
    spl_dict = {"train": [],
                "dev": [],
                "test": []}

    for split in inputs_list_split:
        name_2, split = split.split("|")
        split_dict[name_2] = split
        spl_dict[split].append(name_2)

    info_dict_2 = dict()
    info_dict_2['prefix'] = anno_path_2.replace("sentence_label/video_map.txt", "sentence/frames_256x256px")
    # print(f"Generate information dict from {anno_path}")
    index_id=0
    for file_idx, file_info in tqdm(enumerate(inputs_list_2), total=len(inputs_list_2)):
        # fileid, folder, signer, label = file_info.split("|")
        index, name, length, gloss, char, word, postag = file_info.split("|")

        # num_frames_2 = len(glob.glob(f"{info_dict_2['prefix']}/{name}/*.jpg")) #num_frames = len(glob.glob(f"{info_dict['prefix']}/{dataset_type}/{folder}"))
        try:
            a = split_dict[name]
            if split_dict[name] != dataset_type:
                continue
            # read feature
            feature_root_path = "/home/yejinhui/Projects/SLT/VAC_CSLR/checkpoints/cslt/gloss30check2"
            feature_path = os.path.join(feature_root_path, dataset_type, f"{name}_features.npy")

            features = np.load(feature_path, allow_pickle=True).item()["features"]
        except:
            print(name)
            continue



        info_dict_2[index_id] = {
            'name': dataset_type + "/"+ name,
            'signer': "NONE",
            "gloss": gloss,
            'text': char,
            'sign': torch.tensor(features)
        }
        index_id += 1

    #Save
    features_list = list(info_dict_2.values())
    save_root = "/home/yejinhui/Projects/SLT/data/CSLT/"
    save_path = save_root + f"cslt.word.{dataset_type}" + ".pickle.char"
    with open(save_path, "wb") as list_file:
        pickle.dump(features_list[1:], list_file)

import argparse

import jieba
def char_to_word_level(input_file, output_file):
    with open(input_file, "r", encoding="utf-8") as infile, open(output_file, "w", encoding="utf-8") as outfile:
        for line in infile:
            line = line.strip().replace(" ", "")
            # 使用jieba分词库对文本进行分词
            words = jieba.cut(line, cut_all=False)
            # 将分词后的结果连接起来，用空格分隔
            word_level_line = " ".join(words)
            outfile.write(word_level_line + "\n")


def swap_generated_gloss_data_build(path=None):
    embedded_file= "/home/yejinhui/Projects/SLT/data/CSLT/cslt.word.test.pickle"
    teacher_target_file = "/home/yejinhui/Projects/SLT/training_task/081_cslt_char_sign_ratio0.6_baseline_S2T_seed56_bsz64_drop15_len30_freq100_ratio_1_b4_20_5/gls/30900.dev.hyp.gls"
    teacher_target_file_list = []

    teacher_target_file_list.append(teacher_target_file)


    raw_datas = load_dataset_file(filename=embedded_file)

    signs_len = len(raw_datas)
    for index in range(len(teacher_target_file_list)):
        teacher_target_file = teacher_target_file_list[index]
        teacher_target = read_all_dataset(filename=teacher_target_file)
        # assert signs_len == len(teacher_target) and len(teacher_target) == len(target)
        print(teacher_target_file)
        offset = 0
        for index in range(signs_len):
            raw_name = raw_datas[index]["name"]

            generated_name, gloss = teacher_target[index].split("|")

            if raw_name != generated_name:
                print(generated_name)

            raw_datas[index]["gloss"] = gloss

    save_path = embedded_file + ".generatedGloss"
    with open(save_path, "wb") as list_file:
        pickle.dump(raw_datas, list_file)


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
    #
    # main(data_path=args.data_root_path, tip_str="train", output_path=args.output_path)
    swap_generated_gloss_data_build()
    # get_new_sign_featrue()
    # teacher_model_data_build()
    # build_cslt_sign_featrue(dataset_type="test")
    #
    # build_cslt_sign_featrue(dataset_type="train")
    #
    # build_cslt_sign_featrue(dataset_type="dev")

    # input_file = "/home/yejinhui/Projects/SLT/data/CSLT/PGen/Temp/temp.220812.zh.unique"
    # output_file = "/home/yejinhui/Projects/SLT/data/CSLT/PGen/Temp/temp.220812.zh.unique.word"
    # char_to_word_level(input_file, output_file)




