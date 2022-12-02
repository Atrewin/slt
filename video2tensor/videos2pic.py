from singleVideo2pic import video_to_frames
import os
import threading


def videos_to_frames(index, mode):
    '''
    mode = 'train' or 'dev' or 'test'
    '''
    input_dir = r'/home/yejinhui/Projects/SP-10/videos'  # 输入的video文件夹位置
    save_dir = r'/home/yejinhui/Projects/SP-10/videos'  # 输出目录
    count = 0

    input_dir_final = input_dir + '/' + mode + '/' + str(index)
    save_dir_final = save_dir + '/' + mode + '_Preprocess' + '/' + str(index)
    for video_name in os.listdir(input_dir_final):
        video_path = os.path.join(input_dir_final, video_name)
        outPutDirName = os.path.join(save_dir_final, video_name[:-4])
        threading.Thread(target=video_to_frames, args=(video_path, outPutDirName)).start()
        count = count + 1
        print("%s th video has been finished!" % count)
        print("                ")


if __name__ == '__main__':
    train_dir = r'/home/yejinhui/Projects/SP-10/videos/train'  # 输入的video文件夹位置
    dev_dir = r'/home/yejinhui/Projects/SP-10/videos/dev'
    test_dir = r'/home/yejinhui/Projects/SP-10/videos/test'

    save_dir = r'/home/yejinhui/Projects/SP-10/videos'

    # for index in os.listdir(train_dir):
    #    videos_to_frames(index, 'train')
    for index in os.listdir(dev_dir):
        videos_to_frames(index, 'dev')
    for index in os.listdir(test_dir):
        videos_to_frames(index, 'test')




