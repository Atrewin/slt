cd /home/yejinhui/Projects/SLT/signjoey/external_metrics


hyp=/home/yejinhui/Projects/SLT/training_task/Reading/00_SMKD_sign_mixup_MKD_S2T_seed32_bsz128_drop15_len30_freq50/best.IT_00009850.BW_010.dev.txt
ref=/home/yejinhui/Projects/SLT/training_task/Reading/00_SMKD_sign_mixup_MKD_S2T_seed32_bsz128_drop15_len30_freq50/references.dev.gls

python bleu.py 1 $hyp $ref
python bleu.py 2 $hyp $ref
python bleu.py 3 $hyp $ref
python bleu.py 4 $hyp $ref

