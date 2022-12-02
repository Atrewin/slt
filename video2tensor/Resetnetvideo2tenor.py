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
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Load the pretrained model
resnet152 = models.resnet152(pretrained=True)
resnet152.eval()
resnet152.to(DEVICE)

# Block fc layer
class Identity(nn.Module):
	@overrides
	def forward(self, input_):
		return input_

resnet152.fc = Identity()

# Image transforms
transf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def get_frames(frames_folder_path):
	# Get all frame file names
	frames = None
	frames_file = os.listdir(frames_folder_path)
	for i,frame_file_name in enumerate(frames_file):
		frame = Image.open(os.path.join(frames_folder_path, frame_file_name))
		frame = transf(frame)

		if frames is None:
			frames = torch.empty((len(frames_file), *frame.size()))
		frames[i] = frame

	return frames


def frames_features(frames_folder_path):
	frames = get_frames(frames_folder_path)
	frames = frames.to(DEVICE)
	# Run the model on input data
	output = []
	batch_size = 30                 # 10 for PC
	for start_index in range(0, len(frames), batch_size):
		end_index = min(start_index + batch_size, len(frames))
		frame_range = range(start_index, end_index)
		frame_batch = frames[frame_range]
		avg_pool_value = resnet152(frame_batch)
		output.append(avg_pool_value.detach().cpu().numpy())

	output = np.concatenate(output)

	return output


def videos_features(frames_folders_path, videos_path, save_path):
	# frames_folders_path: path to all video frames folders
	# video_path: path ot original videos
	frames_folders = os.listdir(frames_folders_path)
	features = {}
	for frames_folder in tqdm(frames_folders, ncols=100, ascii=True):
		video_feature = {}
		video_name = os.path.join(videos_path, frames_folder + '.mp4')
		frames_folder_path = os.path.join(frames_folders_path, frames_folder)
		cam = cv2.VideoCapture(video_name)
		fps = round(cam.get(cv2.CAP_PROP_FPS), 0)
		feat = frames_features(frames_folder_path)
		video_feature['fps'] = fps
		video_feature['resnet152'] = feat
		features[frames_folder] = video_feature
		#print("Process video {} FPS {} shape {}".format(frames_folder, fps, feat.shape))

	with open(save_path, 'wb') as f:
		pickle.dump(features, f)


def main():
	frames_folders_path = "./Frames_folders"
	videos_path = "./Videos"
	videos_features(frames_folders_path, videos_path, 'resnet_video_features.pt')


from helpers import *
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
    # split_name = "train"
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
            text= row_list[6]

            video_frames_path = os.path.join(video_root_path, name)

            sign = frames_features(video_frames_path)
            # sign = sign.reshape([sign.shape[0], -1])
            sign = torch.tensor(sign)
            sample_format = {
                "name": name,
                "signer": signer,
                "gloss": gloss,
                "text": text,
                "sign": sign
            }
            answers.append(sample_format)
            index += 1
            if index % 1000 == 0 and len(answers) > 0:
                save_name = split_name + str(start_index) + "_Ph4TResetnet_" + str(index) + ".pickle"
                save_path = os.path.join(output_path, save_name)
                with open(save_path, "wb") as list_file:
                    pickle.dump(answers, list_file)
                del answers
                answers = []
            print(index)

            if index == start_index + 1000:
                break


        except:

            # 封装格式
            if len(answers) > 0:
                save_name = split_name + str(start_index) + "_Ph4TResetnet_" + str(index) + ".pickle"
                save_path = os.path.join(output_path, save_name)
                with open(save_path, "wb") as list_file:
                    pickle.dump(answers, list_file)

            print(index)
            del answers
            answers = []
            pass
            continue




    pass

    # 获取 sign tensor

    # 封装格式
    save_name = split_name + str(start_index) + "_Ph4TResetnet_" + str(index) + ".pickle"
    save_path = os.path.join(output_path,save_name)
    with open(save_path, "wb") as list_file:

        pickle.dump(answers, list_file)

    return None


import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser("Joey-NMT")
    parser.add_argument(
        "--csv_file",
        default="/home/yejinhui/Dataset/PHOENIX-2014-T/annotations/manual/PHOENIX-2014-T.train.corpus.csv",
        type=str,
        help="Training configuration file (yaml).",
    )

    parser.add_argument(
        "--video_root_path",
        default="/home/yejinhui/Dataset/PHOENIX-2014-T/features/fullFrame-210x260px/",
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
        default="./datas/",
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
