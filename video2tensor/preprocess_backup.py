# coding: utf-8
"""
Data module
"""
from torchtext import data
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

from efficientnet_pytorch import EfficientNet
import cv2

from video2tensor import sign2tensor
from json2text import json2text, dictList2dict

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
    train_text_path = '/home/gzs/baseline/sp-10/dataset/train.json' 
    dev_text_path = '/home/gzs/baseline/sp-10/dataset/dev.json'
    test_text_path = '/home/gzs/baseline/sp-10/dataset/test.json'

    for video_index in os.listdir(file_name):
        annotation_file = file_name + '/' + video_index
        for sign_lang_tag in os.listdir(annotation_file):
            video_frames_path = annotation_file + '/' + sign_lang_tag
            
            print("----------\n")
            print("index %s" % video_index)
            print("lang  %s" % sign_lang_tag)

            feature_tensor = sign2tensor(video_frames_path)
            text = json2text(train_text_path, int(video_index), sign_lang_tag) 
            
            print("text : %s" % text)
            print("feature tensor:")
            print(feature_tensor)
            print("----------\n")


if __name__ == '__main__':
    #path = []
    #file = open('train.pickle', 'wb')
    #pickle.dump(path, file)
    #file.close()
    path = '/home/gzs/baseline/sp-10/videos/train_Preprocess'
    preprocess(path)
    '''
    test_file = '/home/gzs/baseline/sp-10/videos/train_afterPreprocess_backup/625/de'
    sign2tensor(test_file)
    train_text_path = '/home/gzs/baseline/sp-10/dataset/train.json'
    text = json2text(train_text_path, 625, 'de')
    print(text)
    '''

'''
def sign2tensor(video_path):
    
    function: obtain feature file of one img by efficientnet
 
    
    imgs = []
    model = EfficientNet.from_pretrained('efficientnet-b0')
    
    
    tfms = transforms.Compose([
        transforms.Resize(600),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])
    
    for image_index in os.listdir(video_path):
        img_path = video_path + '/' +image_index
        img = tfms(Image.open(img_path)).unsqueeze(0) # efficientnet-b0 input: 4 dim
        feature_tensor = model.extract_features(img)
        #print(feature_tensor)
        feature_np = feature_tensor.detach().numpy()
        #print(feature_np)
        imgs.append(feature_np)
    
    # using the simplest way(average) to from video tensor from  many frame pictures tensor
    # later, we can add more information like optical tensor、facial tensor to form one better video tensor
    imgs = np.array(imgs)
    imgs = np.mean(imgs, 0)
    
    features = torch.Tensor(imgs)
    print("video tensor is :")
    print(features.size())
    print(features)

    return features
'''
