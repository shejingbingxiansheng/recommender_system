from model import LFM
from model.sample import Corpus

if __name__ == '__main__':
    # user_id = '151'
    # Corpus().pre_process()
    lfm = LFM()
    lfm.train()
    # import os
    # print(os.path.abspath(os.path.dirname(os.getcwd()) + os.path.sep + "."))