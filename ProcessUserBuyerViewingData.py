import numpy as np
import time
import os
import glob
from PIL import Image, ImageDraw,ImageFont
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn import linear_model
import pickle
import os.path
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
import scipy
from sklearn import preprocessing
from sklearn.preprocessing import Imputer
cwd = os.getcwd()
filepath =cwd+"/user_buy_after_view.csv"
filepath2 =cwd+"/user_notbuyafterview_sample.csv"
df = pd.read_csv(filepath)
df2 = pd.read_csv(filepath2)
df_combine= df.append(df2)
df_combine= df_combine[["item_id" , "brand", "price" , "user_also_view","user_buy_after_viewing"]]

df_combine['user_buy_after_viewing_bool']=0
#df_combine.loc[ (df_combine['user_buy_after_viewing'].isnull()),"user_buy_after_viewing_bool"]=0
df_combine.loc[ (df_combine['user_buy_after_viewing'].notnull()),"user_buy_after_viewing_bool"]=1
df_combine= df_combine[["item_id" , "brand", "price" , "user_also_view","user_buy_after_viewing_bool"]]
df_combine.loc[ (df_combine['brand'].isnull()),"brand"]="No Brand"
#scipy.dense.csr_matrix(df_combine.values)

#X = [[0, 0], [1, 1]]
#y = [0, 1]
df_combine['user_also_view'].replace(regex=True,inplace=True,to_replace=r"\'",value=r'')

le = preprocessing.LabelEncoder()
le.fit(df_combine["item_id"].values )
df_combine["item_id_int"]=0
le2=preprocessing.LabelEncoder()
le2.fit( df_combine["user_also_view"].values)
df_combine["user_also_view_int"]=0
le3=preprocessing.LabelEncoder()
le3.fit( df_combine[(df_combine["brand"].notnull() )]["brand"].values)
df_combine["brand_int"]=0

for eclas in list(le.classes_):
    int= le.transform([eclas])
    df_combine.loc[ (df_combine["item_id"]==eclas ),"item_id_int"] =int[0]

for eclas in list(le2.classes_):
    int= le2.transform([eclas])
    df_combine.loc[ (df_combine["user_also_view"]==eclas ),"user_also_view_int"] =int[0]

for eclas in list(le3.classes_):
    int= le3.transform([eclas])
    df_combine.loc[ (df_combine["brand"]==eclas ),"brand_int"] =int[0]

print(df_combine  )
'''
for index, row in df_combine.iterrows():
    #print(row[ "item_id"])
    item_id=le.transform( [row[ "item_id"]])
    itemidloc= df_combine.columns.get_loc("item_id_int")
    df_combine.iloc[index,itemidloc]  =  item_id

    user_also_view_int=le2.transform( [row[ "user_also_view"]])
    user_also_viewloc= df_combine.columns.get_loc("user_also_view_int")
    df_combine.iloc[index,user_also_viewloc]  =  user_also_view_int
    #print(df_combine)
    #df_combine.iloc[index,0]=
    #print (item_id)
       '''
df_combine.to_csv(cwd+"/user_item_combine.csv", encoding='utf-8',index=False)
#imp = Imputer(missing_values='NaN', strategy='mean', axis=0)

#imp = imp.fit(X_train)

# Impute our data, then train
# print(df_combine[(df_combine["item_id"]=='B0000016MI' )]   )
#X_train_imp = imp.transform(X_train)
#clf = svm.SVC()
#clf.fit( df_combine[["item_id" , "brand", "price" , "user_also_view"]].values,df_combine[["user_buy_after_viewing_bool"]].values)
