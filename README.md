# Sign Language Transformers (CVPR'20)

This repo contains the training and evaluation code for the paper [Sign Language Transformers: Sign Language Transformers: Joint End-to-end Sign Language Recognition and Translation](https://www.cihancamgoz.com/pub/camgoz2020cvpr.pdf). 

This code is based on [Joey NMT](https://github.com/joeynmt/joeynmt) but modified to realize joint continuous sign language recognition and translation. For text-to-text translation experiments, you can use the original Joey NMT framework.
 
## Requirements
* Download the feature files using the `data/download.sh` script.

* [Optional] Create a conda or python virtual environment.

* Install required packages using the `requirements.txt` file.

    `pip install -r requirements.txt`

## Usage

  `python -m signjoey train configs/sign.yaml` 

! Note that the default data directory is `./data`. If you download them to somewhere else, you need to update the `data_path` parameters in your config file.   
## ToDo:

- [X] *Initial code release.*
- [X] *Release image features for Phoenix2014T.*
- [ ] Share extensive qualitative and quantitative results & config files to generate them.
- [ ] (Nice to have) - Guide to set up conda environment and docker image.

## Reference

Please cite the paper below if you use this code in your research:

    @inproceedings{camgoz2020sign,
      author = {Necati Cihan Camgoz and Oscar Koller and Simon Hadfield and Richard Bowden},
      title = {Sign Language Transformers: Joint End-to-end Sign Language Recognition and Translation},
      booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
      year = {2020}
    }

## Acknowledgements
<sub>This work was funded by the SNSF Sinergia project "Scalable Multimodal Sign Language Technology for Sign Language Learning and Assessment" (SMILE) grant agreement number CRSII2 160811 and the European Union’s Horizon2020 research and innovation programme under grant agreement no. 762021 (Content4All). This work reflects only the author’s view and the Commission is not responsible for any use that may be made of the information it contains. We would also like to thank NVIDIA Corporation for their GPU grant. </sub>



aa/fe/b11a9d0e8699dfe94dba211265805b0ea8f4188bf175be8665110fb9f5e4


pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn torch==1.4.0 torchvision==0.5.0

cd /apdcephfs/share_916081/shared_info/zhengshguo/jinhui/Projects/SLT/video2tensor

python video2tensor.py --start_index 0000 \
        --csv_file /apdcephfs/share_916081/shared_info/zhengshguo/jinhui/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/annotations/manual/PHOENIX-2014-T.train.corpus.csv \
        --video_root_path /apdcephfs/share_916081/shared_info/zhengshguo/jinhui/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/features/fullFrame-210x260px \
        --split_name train \
        --output_path /apdcephfs/share_916081/shared_info/zhengshguo/jinhui/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/features


export PATH=/apdcephfs/share_916081/shared_info/zhengshguo/jinhui/anaconda3_old/envs/SLT_1116/bin/:$PATH
cd /apdcephfs/share_916081/shared_info/zhengshguo/jinhui/Projects/SLT

python signjoey/training.py --config /apdcephfs/share_916081/shared_info/zhengshguo/jinhui/Projects/SLT/configs/jizhi/default.yaml


python signjoey train configs/sign.yaml


--start_index 0000 --csv_file /apdcephfs/share_916081/shared_info/zhengshguo/jinhui/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/annotations/manual/PHOENIX-2014-T.test.corpus.csv --video_root_path /apdcephfs/share_916081/shared_info/zhengshguo/jinhui/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/features/fullFrame-210x260px --split_name test --output_path /apdcephfs/share_916081/shared_info/zhengshguo/jinhui/Projects/SLT/data/RetsetNet

