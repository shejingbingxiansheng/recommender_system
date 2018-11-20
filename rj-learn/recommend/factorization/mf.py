import pandas as pd
import numpy as np
import pickle
import os
class MF:
    def __init__(self,k,epoch,lr,lam,dataFrame):
        '''
        :param k: 隐因子数量，一般在50~200
        :param epoch: 迭代轮数
        :param lr: 学习率
        :param lam: 正则化系数
        '''
        self.k = k
        self.epoch = epoch
        self.lr = lr
        self.lam = lam
        self.frame = dataFrame
        self.init_model()


    def init_model(self):
        self.user_ids = set(self.frame['UserId'].values)
        self.item_ids = set(self.frame['MovieId'].values)
        self._get_user_items_dict()
        #初始化P,Q矩阵
        array_P = np.random.randn(len(self.user_ids),self.k)
        array_Q = np.random.randn(len(self.item_ids),self.k)

        #将矩阵转化为df
        self.P = pd.DataFrame(array_P,index=list(self.user_ids),columns=range(0,self.k))
        self.Q = pd.DataFrame(array_Q,index=list(self.item_ids),columns=range(0,self.k))


    def _get_user_items_dict(self):
        '''
        构建用户的评分字典，格式{userId1:{MovieId1:2,MovieId2:5,....}}
        :return:
        '''
        user_items_dict = {}
        for user_id in self.user_ids:
            items_rating = list(self.frame[self.frame['UserId']==user_id][['MovieId','Rating']].values)
            item_rating_dict = {}
            for item_rating in items_rating:
                # item_id = item_rating[0]
                # r = item_rating[1]
                item_rating_dict[item_rating[0]] = item_rating[1]
            user_items_dict[user_id] = item_rating_dict
        self.user_items_dict = user_items_dict

    def _predict(self,user_id,item_id):
        p_i = np.mat(self.P.ix[user_id].values)
        q_j = np.mat(self.Q.ix[item_id].values).T
        r = (p_i*q_j).sum()
        # if r>100:
        #     return 5
        # elif r<-100:
        #     return 1
        return r
    def _loss(self,epoch,user_id,item_id,y):
        r = self._predict(user_id,item_id)
        # print('p_i: {},q_j: {}'.format(self.P.ix[user_id].values,self.Q.ix[item_id].values))
        # p_i = np.mat(self.P.ix[user_id].values)
        # q_j = np.mat(self.Q.ix[item_id].values).T
        e = y-r
        loss = e**2
        print('Epoch: {},user_id: {},item_id: {},y: {},r: {},e: {},loss:{}'.format(epoch,user_id,item_id,y,r,e,loss))
        return e
    def _optimization(self,user_id,item_id,e):
        gradient_p = (-1*self.Q.ix[item_id].values)*e+self.lam*self.P.ix[user_id].values
        gradient_q = (-1*self.P.ix[user_id].values)*e+self.lam*self.Q.ix[item_id].values
        # print('gradient_p: {},gradient_q: {}'.format(gradient_p,gradient_q))
        self.P.loc[user_id] -= self.lr*gradient_p
        self.Q.loc[item_id] -= self.lr*gradient_q
    def train(self):
        for step in range(0,self.epoch):
            for user_id,items_dict in self.user_items_dict.items():
                item_ids = list(items_dict.keys())
                for item_id in item_ids:
                    e = self._loss(step,user_id,item_id,items_dict[item_id])
                    # print('p_i: {},q_j: {}'.format(self.P.ix[user_id].values, self.Q.ix[item_id].values))
                    self._optimization(user_id,item_id,e)
            self.lr *= 0.9
        self.save()

    def predict(self,user_id,top_n):
        self.load()
        # 指定用户看过的item
        user_item_ids = set(self.user_items_dict[user_id].keys())
        # 拿到用户没看过的item
        other_item_ids = self.item_ids^user_item_ids
        # 预测用户没看过的item的偏好程度
        pref = [self._predict(user_id,item_id) for item_id in other_item_ids]
        other_item_pref = zip(list(other_item_ids),pref)
        # 关于值排序，从大到小
        order_item_pref = sorted(other_item_pref,key=lambda x:x[1],reverse=True)
        #取前topN
        return order_item_pref[:top_n]

    def save(self):
        f = open(os.getcwd()+"/FM.factorization",'wb')
        pickle.dump((self.P,self.Q),f)
        f.close()

    def load(self):
        f = open(os.getcwd()+"/FM.factorization",'rb')
        self.P,self.Q = pickle.load(f)
        f.close()
