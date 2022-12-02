while true
do
  continue
done
#
#

export PATH=/apdcephfs/share_916081/jinhuiye/Environments/anaconda3/envs/SLT/bin/:$PATH

#export PATH=/apdcephfs/share_916081/jinhuiye/Environments/Python/python3.7.2/bin/:$PATH
#cd /jizhi/jizhi2/worker/trainer/ZhEn/pytorch-nmt-master

#python -m signjoey train configs/sign.yaml
cd /apdcephfs/share_916081/jinhuiye/Projects/slt
python -m signjoey train /jizhi/jizhi2/worker/trainer/sign.yaml