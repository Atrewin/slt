def add_project_root():
    import sys
    from os.path import abspath, join, dirname
    sys.path.insert(0, abspath(join(abspath(dirname(__file__)), '../')))


add_project_root()
from signjoey.prediction import *
from helpers import *

from numpy import *
import matplotlib.pyplot as plt

class Analyze():
    def __init__(self):
        pass

    def length(self):
        #parm


        data_neigbor_add = ""
        neighbor_decrease_frames = "/apdcephfs/share_916081/jinhuiye/Projects/slt/training_task/sign_model_seed_7/output/neighbor_decrease_frames/BW_02.A_-1.test.txt"


    def calculate_sentence_bleu(self, inference, reference):
        from nltk.translate.bleu_score import sentence_bleu

        ref = []
        cand = []
        ref.append(reference.split())
        cand = (inference.split())

        score_1_4_gram = sentence_bleu(ref, cand, weights=(0, 0, 0, 1))

        return score_1_4_gram

    def MP_BLEU_Analyze(self, raw_output_file, MP_output_file, Reference_file, result_path="/apdcephfs/share_916081/jinhuiye/Projects/slt/training_task/sign_model_seed_7/output"):
        raw_output_sentence = read_all_dataset(raw_output_file)
        MP_output_sentence = read_all_dataset(MP_output_file)
        Reference_file_sentence = read_all_dataset(Reference_file)

        assert len(raw_output_sentence) == len(MP_output_sentence) and len(MP_output_sentence) == len(Reference_file_sentence)
        BLEU_raw = []
        BLEU_MP = []

        for index in range(len(Reference_file_sentence)):
            reference = Reference_file_sentence[index].split("|")[-1]
            inference_raw = raw_output_sentence[index].split("|")[-1]
            inference_raw_mp = MP_output_sentence[index].split("|")[-1]
            BLEU_raw.append(self.calculate_sentence_bleu(inference=inference_raw, reference=reference))
            BLEU_MP.append(self.calculate_sentence_bleu(inference=inference_raw_mp, reference=reference))

        # writer2text(data_rows=BLEU_raw,file_path=os.path.join(result_path, "BLEU_raw.txt"))
        # writer2text(data_rows=BLEU_MP,file_path=os.path.join(result_path, "BLEU_MP.txt"))

        writer2text(data_rows=BLEU_raw, file_path=raw_output_file+".BLEU.txt")
        writer2text(data_rows=BLEU_MP, file_path=MP_output_file+".BLEU.txt")
        self.fig(X=BLEU_raw, Y=BLEU_MP)
        pass
        print("##" * 10)

    def fig(self, X, Y):

        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        # 设置标题
        ax1.set_title('Scatter Plot')
        # 设置X轴标签
        plt.xlabel('Transformer BLEU-4')
        # 设置Y轴标签
        plt.ylabel('mT5 BLEU-4')
        # 画散点图
        ax1.scatter(x=X, y=Y, c='r', marker='o')
        # 设置图标
        # plt.legend('x1')
        # 显示所画的图
        plt.show()

    def get_length_buket(self, cops):
        data_row = read_all_dataset(cops)

        dict_len = {}
        total=0

        for sentencet in data_row:
            leng = len(sentencet.split(" "))

            if leng in dict_len.keys():
                dict_len[leng] += 1
            else:
                dict_len[leng] = 1

            total += 1

        ans = {
            1: 0,
            2: 0,
            3: 0,
            4: 0
        }

        for key in dict_len.keys():
            buk = dict_len[key]

            if key < 10:
                ans[1] += buk
            elif key < 20:
                ans[2] += buk
            # elif key < 30 :
            #     ans[3] += buk
            # elif key < 40 :
            #     ans[4] += buk

            # print(key, "\t", buk)
            elif key < 3000 :
                ans[3] += buk


        for key in ans.keys():
            buk = ans[key]


            print(key, "\t", buk)
if __name__ == "__main__":

    Analyzer = Analyze()

    raw_output_file = "/apdcephfs/share_916081/jinhuiye/Projects/slt/training_task/sign_model_seed_7/output/raw_sign/BW_02.A_-1.test.txt"
    MP_output_file = "/apdcephfs/share_916081/jinhuiye/Projects/slt/training_task/sign_model_seed_7/output/ramdom_repeated_interval/BW_02.A_-1.test.txt"
    Reference_file = "/apdcephfs/share_916081/jinhuiye/Projects/slt/training_task/DE.test.txt"


    raw_output_file = "/apdcephfs/share_916081/jinhuiye/Temp/Gloss_to_Text/Exp/phoenix2014T/results/test_output/de_xx/decoding.txt"
    MP_output_file = "/apdcephfs/share_916081/jinhuiye/Projects/GPT2/Data/Sign/phoenix2014T.test.de.mt5.gloss"
    Reference_file = "/apdcephfs/share_916081/jinhuiye/Projects/GPT2/Data/Sign/phoenix2014T.test.gloss.lower"

    #中文
    # raw_output_file = "/apdcephfs/share_916081/jinhuiye/Temp/Gloss_to_Text/Exp/CSL/results/Test/test/decoding.txt"
    # MP_output_file = "/apdcephfs/share_916081/jinhuiye/Projects/GPT2/Data/Sign/Zh/csl-daily.label_char.test.mT5.gloss"
    # Reference_file = "/apdcephfs/share_916081/jinhuiye/Projects/GPT2/Data/Sign/Zh/csl-daily.label_gloss.test"
    # raw_output_file = "/apdcephfs/share_916081/jinhuiye/Temp/Gloss_to_Text/Exp/CSL/data/rawdata/tempdata_2/trans.de.txt"
    # MP_output_file = "/apdcephfs/share_916081/jinhuiye/Temp/Gloss_to_Text/Exp/CSL/data/rawdata/tempdata_2/trains.best.gpt40.de.txt"
    # Reference_file = "/apdcephfs/share_916081/jinhuiye/Temp/Gloss_to_Text/Exp/CSL/data/rawdata/tempdata_2/target.txt"
    #
    # gloss-text

    raw_output_file = "/apdcephfs/share_916081/jinhuiye/Temp/Gloss_to_Text/phoenix2014T.data_word/phoenix2014T.dev.de"
    MP_output_file = "/apdcephfs/share_916081/jinhuiye/Temp/Gloss_to_Text/Exp/phoenix2014T/results/test_output/xx_de_11/valid/decoding.txt"
    Reference_file = "/apdcephfs/share_916081/jinhuiye/Temp/Gloss_to_Text/Exp/phoenix2014T/results/test_output/xx_de_G30w_T5_tag_finetuning_111/valid/decoding.txt"

    # Analyzer.MP_BLEU_Analyze(raw_output_file=raw_output_file, MP_output_file=MP_output_file, Reference_file=Reference_file)

    gpt40 = "/apdcephfs/share_916081/jinhuiye/Temp/Gloss_to_Text/phoenix2014T.data_word/generating_data/find_best_times/need2.80w.de.clear.40Times.de"
    Analyzer.get_length_buket(cops=gpt40)