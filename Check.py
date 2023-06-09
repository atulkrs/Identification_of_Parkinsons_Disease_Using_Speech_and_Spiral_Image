import os
import pandas as pd
path='spiralimages/'
dataset=[]
for folder in os.listdir(path):
    for file in os.listdir(os.path.join(path,folder)):
        print(os.path.join(path,folder,file))
        temp=[]
        temp.append(os.path.join(path,folder,file))
        temp.append(folder)
        dataset.append(temp)
df=pd.DataFrame(dataset)
df.to_csv('Imageinfo2.csv')
