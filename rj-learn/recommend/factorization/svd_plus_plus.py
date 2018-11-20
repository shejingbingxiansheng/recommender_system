import pandas as pd
import numpy as np

class SVDPlusPlus:
    def __init__(self,k,epoch,lr,lam):
        self.k=k
        self.epoch = epoch
        self.lr=lr
        self.lam=lam

    def init_model(self):
        print()
