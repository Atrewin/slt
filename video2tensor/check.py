import os
from singleVideo2pic import video_to_frames_all_lan
import shutil

def checkHavepics(path):
    BlankDirList = []
    for index in os.listdir(path):
        path_2 = path + '/' + str(index)
        #print(path_2)
        for lang in os.listdir(path_2):
            path_3 = path_2 + '/' + str(lang)
            if len(os.listdir(path_3))==0:
                print("%s" % index)
                BlankDirList.append(index)
                break
    return BlankDirList 

def deleteDir(path):
    shutil.rmtree(path)

def repairDir(indexList, input_path, output_path):

    for index in indexList:
    #with open(index_file, 'r') as f:
        #for index in f:
            delete_path = output_path + '/' + index # [:-1]
            inputPath = input_path + '/' + index # [:-1]

            deleteDir(delete_path)
            video_to_frames_all_lan(inputPath, delete_path)


if __name__=='__main__':
    l = []
    input_path = '/home/gzs/baseline/sp-10/videos/train'
    path = '/home/gzs/baseline/sp-10/videos/train_Preprocess'
    
    index_file = '/home/gzs/baseline/videoPreprocess/check.list'
    l = checkHavepics(path)
    
    repairDir(l, input_path, path)
    '''    
    for index in l:
    #with open(index_file, 'r') as f:
        #for index in f:
            delete_path = path + '/' + index # [:-1]
            inputPath = input_path + '/' + index # [:-1]

            deleteDir(delete_path)
            video_to_frames_all_lan(inputPath, delete_path)
    '''
