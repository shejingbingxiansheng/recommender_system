import pandas as pd
import numpy as np

class SVDPlus:
    def __init__(self,k,epoch,lr,lam):
        self.k=k
        self.epoch = epoch
        self.lr=lr
        self.lam=lam

    def init_model(self):
        ratings_length = self.frame.shape[0]#评分数量
        total_ratings = self.frame[['Rating']].sum()#总评分数
        self.μ = float(total_ratings / ratings_length)  # 所有评分的总平均值
        self.user_ids = set(self.frame[['UserId']].values)
        self.item_ids = set(self.frame[['MovieId']].values)
        self._get_user_item_ratings()
        self.user_count = len(self.user_ids)
        self.item_count = len(self.item_ids)
        array_P = np.random.randn(self.user_count,self.k)
        array_Q = np.random.randn(self.item_count,self.k)
        self.P = pd.DataFrame(array_P,index=list(self.user_ids),columns=range(0,self.k))
        self.Q = pd.DataFrame(array_Q,index=list(self.item_ids),columns=range(0,self.k))
        array_Bu = np.random.randn(self.user_count,1)
        array_Bi = np.random.randn(self.item_count,1)
        self.Bu = pd.DataFrame(array_Bu,index=list(self.user_ids),columns=[0])
        self.Bi = pd.DataFrame(array_Bi,index=list(self.item_ids),columns=[0])
        arrya_y = np.random.randn(self.item_count,self.k)
        self.y = pd.DataFrame(arrya_y,index=list(self.item_ids),columns=range(0,self.k))


    def _get_user_item_ratings(self):
        '''
        构建用户的评分字典，格式{userId1:{MovieId1:2,MovieId2:5,....}}
        '''
        #保存每个用户评分的电影数量
        array_Ru = np.zeros((self.user_count,1))
        self.Ru = pd.DataFrame(array_Ru,index=list(self.user_count),columns=[0])

        user_item_rat = {}
        for user_id in self.user_ids:
            items_rat_list = list(self.frame[self.frame['UserId']==user_id][['MovieId','Rating']].values)
            self.Ru.loc[user_id,0]=len(items_rat_list)
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
        ru = self.Ru.ix[user_id].values
        #获取该用户的评分item列表
        item_ids = list(self.user_item_rat[user_id].keys())
        yis = self.y[[item_ids]].values
        yi = yis[0]
        for i in range(1,yi.shape[0]):
            yi+=yis[i]
        pred = self.μ+bi+bu+((p+yi/(ru**0.5))*q).sum()
        return pred

    def _error(self,step,user_id,item_id,rui):
        e = rui-self._predict(user_id,item_id)
        loss = e ** 2
        print('Step:{}, user_id:{}, item_id:{}, loss:{}'.format(step, user_id, item_id, loss))
        return e

    def _optimization(self,user_id,item_id,e):
        gradient_bu = -1*e+self.lam*self.Bu.ix[user_id].values
        gradient_bi = -1*e+self.lam*self.Bi.ix[item_id].values
        gradient_p = (-1*self.Q.ix[item_id].values)*e+self.lam*self.P.ix[user_id].values
        ru = self.Ru.ix[user_id].values
        item_ids = list(self.user_item_rat[user_id].keys())
        yis = self.y[[item_ids]].values
        yi = yis[0]
        for i in range(1, yi.shape[0]):
            yi += yis[i]
        gradient_q = -1*(self.P.ix[user_id].values+yi/(ru**0.5))*e+self.lam*self.Q.ix[item_id].values


        self.Bu.loc[user_id] -= self.lr*gradient_bu
        self.Bi.loc[item_id] -= self.lr*gradient_bi
        self.P.loc[user_id] -= self.lr*gradient_p
        self.Q.loc[item_id] -= self.lr*gradient_q
        for id in item_ids:
            gradient_yid = -1*e*(self.Q.ix[id]/(ru**0.5))+self.lam*self.y.ix[id].values
            self.y.loc[id] -= self.lr*gradient_yid
