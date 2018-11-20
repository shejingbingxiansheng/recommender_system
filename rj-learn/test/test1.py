import os
import pandas as pd

# print(os.path.abspath(os.path.dirname(os.getcwd())+os.path.sep+"."))
df = pd.read_csv('E:/python/pyWorkspace/recommender_system/rj-learn/data/ratings.csv')
t=df['UserId'].values
t2 = df[['UserId']].values
print(t)
print(t2)
# df = df[df['UserId']==1]
# items_ratings = list(df[df['UserId']==1][['MovieId','Rating']].values)
# r = items_ratings[0][1]
# print(items_ratings)