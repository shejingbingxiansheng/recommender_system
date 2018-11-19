import pandas as pd
import numpy as np
import os
import pickle

class SVD:
    def __init__(self,k=10,epoch=10,lr=0.005,lam=0.02):
        self.k=k
        self.epoch = epoch
        self.lr=lr
        self.lam = lam

    def init_model(self):
        ratings_length = self.frame.shape[0]
        total_ratings = self.frame[['Rating']].sum()
        self.μ = float(total_ratings/ratings_length)#所有评分的总平均值
        self.user_ids = set(self.frame['UserId'].values)#用户id的集合
        self.item_ids = set(self.frame['MovieId'].values)
        self._get_user_item_ratings()
        self.user_count = len(self.user_ids)#用户总数量
        self.item_count = len(self.item_ids)#电影总数量

        # 初始化P,Q矩阵
        array_P = np.random.randn(self.user_count,self.k)
        array_Q = np.random.randn(self.item_count,self.k)
        self.P = pd.DataFrame(array_P,index=list(self.user_ids),columns=range(0,self.k))
        self.Q = pd.DataFrame(array_Q,index=list(self.item_ids),columns=range(0,self.k))

        #初始化用户，物品的偏置向量
        array_Bu = np.random.randn(self.user_count)
        array_Bi = np.random.randn(self.item_count)
        self.Bu = pd.DataFrame(array_Bu,index=list(self.user_ids),columns=[0])
        self.Bi = pd.DataFrame(array_Bi,index=list(self.item_ids),columns=[0])

    def _get_user_item_ratings(self):
        '''
        构建用户的评分字典，格式{userId1:{MovieId1:2,MovieId2:5,....}}
        '''
        user_item_rat = {}
        for user_id in self.user_ids:
            items_rat_list = list(self.frame[self.frame['UserId']==user_id][['MovieId','Rating']].values)
            item_rat_dict = {}
            for item_rat in items_rat_list:
                item_rat_dict[item_rat[0]] = item_rat[1]
            user_item_rat[user_id] = item_rat_dict
        self.user_item_rat = user_item_rat

    def _predict(self,user_id,item_id):
        p = np.mat(self.P.ix[user_id].values)
        q = np.mat(self.Q.ix[item_id].values).T
        bu = self.Bu.ix[user_id].values
        bi = self.Bi.ix[item_id].values
        r = self.μ+bi+bu+(p*q).sum()
        return r

    def _error(self,step,user_id,item_id,y):
        e = y-self._predict(user_id,item_id)
        loss = e**2
        print('Step:{}, user_id:{}, item_id:{}, loss:{}'.format(step,user_id,item_id,loss))
        return e
    def _optimization(self,user_id,item_id,e):
        gradient_p = (-1*self.Q.ix[item_id].values)*e+self.lam*self.P.ix[user_id]
        gradient_q = (-1*self.P.ix[user_id].values)*e+self.lam*self.Q.ix[item_id]
        gradient_bu = -1*e+self.lam*self.Bu.ix[user_id].values
        gradient_bi = -1*e+self.lam*self.Bi.ix[item_id].values

        self.P.loc[user_id] -= self.lr*gradient_p
        self.Q.loc[item_id] -= self.lr*gradient_q
        self.Bu.loc[user_id] -= self.lr*gradient_bu
        self.Bi.loc[item_id] -= self.lr*gradient_bi

    def fit(self,df):
        self.frame = df
        self.init_model()
        for step in range(0,self.epoch):
            for user_id,item_rat_dict in self.user_item_rat.items():
                item_ids = list(item_rat_dict.keys())
                for item_id in item_ids:
                    e = self._error(step,user_id,item_id,item_rat_dict[item_id])
                    self._optimization(user_id,item_id,e)
            self.lr *= 0.9


    def predict(self,user_id,top_n):
        '''
        为指定用户推荐topn
        :param user_id:
        :param top_n:
        :return:
        '''

        # 拿到指定用户评分过的电影
        user_item_ids = set(self.frame[self.frame['UserId']==user_id][['MovieId']].values)
        #拿到用户没评分过的其他电影
        other_item_ids = self.item_ids^user_item_ids
        pref = [self._predict(user_id,item_id) for item_id in other_item_ids]
        other_item_pref = zip(list(other_item_ids),pref)
        # 按值排序，从大到小
        order_item_pref = sorted(other_item_pref,key=lambda x:x[1],reverse=True)
        return order_item_pref[:top_n]

    def save(self,path):
        f = open(path,'wb')
        pickle.dump(self,f)
        f.close()

    # def load(self):
    #     f = open(os.getcwd()+"/svd.model",'rb')
    #     self.P,self.Q,self.Bu,self.Bi = pickle.load(f)
    #     f.close()
