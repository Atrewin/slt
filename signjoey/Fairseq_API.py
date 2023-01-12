

def get_G2TModel(conf):
    def add_project_root():  # @jinhui
        import sys
        from os.path import abspath, join, dirname
        sys.path.insert(0, abspath("/apdcephfs/share_916081/shared_info/zhengshguo/jinhui/Projects/fairseq"))

    add_project_root()

    import os
    import subprocess

    import torch

    from fairseq import bleu, checkpoint_utils, options, progress_bar, tasks, utils
    from fairseq.meters import StopwatchMeter, TimeMeter

    pass
