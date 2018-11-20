import random
import pickle
import pandas as pd
import numpy as np
from math import exp
from recommend.factorization.sample import Corpus
import os
class LMF:
    '''
    利用隐式反馈的推荐算法
    '''
    def __init__(self,class_count=5,iter_count=5,lr=0.02,lam=0.01):
        '''
        :param class_count: 隐因子数量，即参数k
        :param iter_count: 迭代次数
        :param lr: 学习率
        :param lam: 正则化系数
        '''
        self.class_count = class_count
        self.iter_count = iter_count
        self.lr = lr
        self.lam = lam
        self._init_model()

    def _init_model(self):
        file_path = os.getcwd() + "/data/ratings.csv"
        self.frame = pd.read_csv(file_path)
        self.user_ids = set(self.frame['UserId'].values)
        self.item_ids = set(self.frame['MovieId'].values)
        self.items_dict = Corpus().load()

        # 初始化P,Q矩阵,标准正态分布
        array_p = np.random.randn(len(self.user_ids),self.class_count)
        array_q = np.random.randn(len(self.item_ids),self.class_count)

        # 索引是user_ids,列名是0,1，...,class_count-1
        self.p = pd.DataFrame(array_p,columns=range(0,self.class_count),index=list(self.user_ids))
        # length = self.p.columns.tolist()
        # print('length: {}'.format(length))
        # 索引是item_ids，列名是0,1，...，class_count-1
        self.q = pd.DataFrame(array_q,columns=range(0,self.class_count),index=list(self.item_ids))

    def _predict(self,user_id,item_id):
        '''
        预测指定用户对指定item的评分
        :param user_id:
        :param item_id:
        :return:
        '''
        p_i = np.mat(self.p.ix[user_id].values)
        q_j = np.mat(self.q.ix[item_id].values).T
        r = (p_i*q_j).sum()
        print('r: {}'.format(r))

        if r<-100:
            return 0
        elif r>100:
            return 1
        logit = 1.0 / (1 + exp(-r))
        print('logit: {}'.format(logit))
        return logit
    def _loss(self,user_id,item_id,y,step):
        e = y-self._predict(user_id,item_id)
        print('Step: {},user_id: {},item_id: {},y: {} loss: {}'.format(step,user_id,item_id,y,e))
        return e

    def _optimation(self,user_id,item_id,e):
        # 这里使用SGD进行求解，使用L2正则化
        # 先对p求导
        gradient_p = (-1*self.q.ix[item_id].values)*e+self.lam*self.p.ix[user_id].values
        # 对q求导
        gradient_q = (-1*self.p.ix[user_id].values)*e+self.lam*self.q.ix[item_id].values
        self.p.loc[user_id] -= gradient_p
        self.q.loc[item_id] -= gradient_q

    def train(self):
        for step in range(0,self.iter_count):
            for user_id,item_dict in self.items_dict.items():
                # 取出当前用户对所有item的隐式反馈
                item_ids = list(item_dict.keys())
                random.shuffle(item_ids)
                for item_id in item_ids:
                    e = self._loss(user_id,item_id,item_dict[item_id],step)
                    self._optimization(user_id,item_id,e)

            # 学习率衰减
            self.lr*=0.9
        self.save()

    def predict(self,user_id,top_n=10):
        '''
        返回topN
        :param user_id:
        :param top_n:
        :return:
        '''
        self.load()
        # 拿到用户已看过的电影Id
        user_item_ids = set(self.frame[self.frame['UserId']==user_id]['MovieId'])
        # 拿到用户没看过的电影Id
        other_item_ids = self.item_ids^user_item_ids
        # 对每个没看过的电影预测用户的对它的偏好程度
        preference = [self._predict(user_id,item_id) for item_id in other_item_ids]
        other_item_preference = zip(list(other_item_ids),preference)
        #对上面的字典进行按值从大到小排序
        order_item_pref = sorted(other_item_preference,key=lambda x:x[1],reverse=True)
        #返回前topn个
        return order_item_pref[:top_n]

    def save(self):
        f = open(os.path.abspath(os.path.dirname(os.getcwd()) + os.path.sep + ".") + "/data/lfm.factorization",'wb')
        pickle.dump((self.p,self.q),f)
        f.close()

    def load(self):
        f=open(os.path.abspath(os.path.dirname(os.getcwd()) + os.path.sep + ".") + "/data/lfm.factorization",'rb')
        self.p,self.q = pickle.load(f)
        f.close()