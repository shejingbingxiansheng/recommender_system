import os
import pandas as pd
import pickle

class Corpus:
    items_dict_path = os.getcwd()+"/data/lfm_items.dict"

    @classmethod
    def pre_process(cls):
        file_path = os.getcwd()+"/data/ratings.csv"
        # pd读取评分文件
        cls.frame = pd.read_csv(file_path)
        # 将评分过的UserId，放在set中
        cls.user_ids = set(cls.frame['UserId'].values)
        # 将MovieId也放入set中
        cls.item_ids = set(cls.frame['MovieId'].values)
        # 构建user-item评分字典，格式为{user_id:{item_id0:1,item_id1:0,.....}}
        cls.items_dict = {user_id:cls._get_pos_neg_item(user_id) for user_id in list(cls.user_ids)}
        cls.save()

    @classmethod
    def _get_pos_neg_item(cls,user_id):
        '''
        得到指定用户的正负item
        正item代表用户有过评分的item
        负item代表用户之前没有看过的
        通过下采样的方式解决样本不平衡的问题
        :param user_id: 指定用户id
        :return:
        '''
        #正样本
        pos_item_ids = set(cls.frame[cls.frame['UserId']==user_id]['MovieId'])
        #负样本
        neg_item_ids = cls.item_ids^pos_item_ids
        neg_item_ids = list(neg_item_ids)[:len(neg_item_ids)]
        item_dict = {}
        # 正样本为1，负样本为0
        for item in pos_item_ids: item_dict[item] = 1
        for item in neg_item_ids:item_dict[item] = 0
        return item_dict

    @classmethod
    def save(cls):
        f = open(cls.items_dict_path,'wb')
        pickle.dump(cls.items_dict,f)
        f.close()
    @classmethod
    def load(cls):
        f = open(cls.items_dict_path,'rb')
        items_dict = pickle.load(f)
        f.close()
        return items_dict

