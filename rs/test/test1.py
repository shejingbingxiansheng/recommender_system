import os
import pandas as pd

# print(os.path.abspath(os.path.dirname(os.getcwd())+os.path.sep+"."))
df = pd.read_csv('E:/PyCharm/pythonWorkSpace/recommender_system_git/rs/data/ratings.csv')
df = df[df['UserId']==1]
items_ratings = list(df[df['UserId']==1][['MovieId','Rating']].values)
r = items_ratings[0][1]
print(items_ratings)