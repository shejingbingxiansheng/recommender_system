# 将movielens的dat数据转换为csv的工具类
import pandas as pd
import os

class Dat2csv:
    def __init__(self,origin_path=None):#默认为data目录
        if origin_path==None:
            origin_path = os.path.abspath(os.path.dirname(os.getcwd())+os.path.sep+".")+"/data"
        self.origin_path = origin_path
    def transform(self):
        print("transform movies.dat")
        self.transform_movies()
        print("transform users.dat")
        self.transform_users()
        print("transform ratings.dat")
        self.transform_ratings()

    def transform_movies(self):
        f = pd.read_table(self.origin_path+"/movies.dat",sep="::",engine="python",
                          names=["MovieId","Title","Genres"])
        f.to_csv(self.origin_path+"/movies.csv",index=False)

    def transform_users(self):
        f = pd.read_table(self.origin_path+"/users.dat",sep="::",engine="python",
                          names=["UserId","Gender","Age","Occupation","Zip-code"])
        f.to_csv(self.origin_path+"/users.csv",index=False)

    def transform_ratings(self):
        f = pd.read_table(self.origin_path + "/ratings.dat", sep="::", engine="python",
                          names=["UserId", "MovieId", "Rating", "Timestamp"])
        f.to_csv(self.origin_path + "/ratings.csv", index=False)

if __name__ == '__main__':
    Dat2csv().transform()
