The procedures of preprocess are as follows:
1. Extract frames  pictures from video;
2. Turn frame pictures into frame tensor, future more, into video represntation tensor;
3. Find the corresponding text of video.

The first step is completed by videos2pic.py. To ensure not missing one video dir, we can use check.py to check and repair if mistakes happened.

The second step is completed by video2tensor.py which uses efficientnet to extract features. We released the pytorch version and tensorflow version.

The third step is completed by json2text.py.

Also, you can do th second step and third step in the meantime by preprocess.py.




cd /home/yejinhui/Projects/SLT/video2tensor
conda activate MSLT

CUDA_VISIBLE_DEVICES=3 python Resetnetvideo2tenor.py --start_index 0000 \
        --csv_file /home/yejinhui/Dataset/PHOENIX-2014-T/annotations/manual/PHOENIX-2014-T.train.corpus.csv \
        --video_root_path /home/yejinhui/Dataset/PHOENIX-2014-T/features/fullFrame-210x260px \
        --split_name train \
        --output_path /home/yejinhui/Dataset/PHOENIX-2014-T/features/



--start_index 7000 --csv_file /apdcephfs/share_916081/shared_info/zhengshguo/jinhui/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/annotations/manual/PHOENIX-2014-T.train.corpus.csv --video_root_path /apdcephfs/share_916081/shared_info/zhengshguo/jinhui/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/features/fullFrame-210x260px --split_name train --output_path /apdcephfs/share_916081/shared_info/zhengshguo/jinhui/Projects/SLT/data/EfficentNet
