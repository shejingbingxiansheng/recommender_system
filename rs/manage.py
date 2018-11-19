from model import LFM
from model.sample import Corpus
from model.fm import FM
import pandas as pd
import os

if __name__ == '__main__':
    # user_id = '151'
    # Corpus().pre_process()
    # lfm = LFM()
    # lfm.train()
    # import os
    # print(os.path.abspath(os.path.dirname(os.getcwd()) + os.path.sep + "."))
    df = pd.read_csv(os.getcwd()+"/data/ratings.csv")
    fm = FM(10,10,0.01,0.02,df)
    fm.train()