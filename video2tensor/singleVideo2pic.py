import cv2
import os
import threading

def video_to_frames(video_path, outPutDirName):
    times = 0
    
    # 提取视频的频率，每1帧提取一个
    frame_frequency = 1
    
	# 如果文件目录不存在则创建目录
    if not os.path.exists(outPutDirName):
        os.makedirs(outPutDirName)
        
    # 读取视频帧
    camera = cv2.VideoCapture(video_path)
    
    while True:
        times = times + 1
        res, image = camera.read()
        if not res:
            print('not res , not image')
            break
        if times % frame_frequency == 0:
            print("writing images into %s" % outPutDirName)
            cv2.imwrite(outPutDirName + '/' + str(times)+'.jpg', image) # linux使用/作为文件分割符
            
    print('图片提取结束')
    camera.release()


#if __name__ == "__main__":
    #input_dir = r'/home/gzs/baseline/sp-10/videos/train/420'       # 输入的video文件夹位置
    #save_dir = r'/home/gzs/baseline/sp-10/videos/train_Preprocess/420'         # 输出目录
def video_to_frames_all_lan(input_dir, save_dir):
    count = 0   
    for video_name in os.listdir(input_dir):
        video_path = os.path.join(input_dir, video_name)
        outPutDirName = os.path.join(save_dir, video_name[:-4])
        threading.Thread(target=video_to_frames, args=(video_path, outPutDirName)).start()
        count = count + 1
        print("%s th video has been finished!" % count)
        print("                ")
