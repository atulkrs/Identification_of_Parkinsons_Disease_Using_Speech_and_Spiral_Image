import os
import pandas as pd
from feature_extraction_copy import Feature_Extraction

df = pd.DataFrame
data=[]
# c=0
# path='audioSplit'
# for p in os.listdir('C:/Users/atulk/Downloads/'):
    # if(c>0):break uncomment for one
    # c=c+1
    # fullPath=os.path.join(path,p)
    # print(fullPath)
    # df=Feature_Extraction().extract_mfcc_from_folder(fullPath)
    # print(df.shape)
#     data.append(df.iloc[0].tolist())
#     print(df.iloc[0].tolist())
#     Feature_Extraction().convert_to_csv(data,'MFCCFeatures')
# data=pd.DataFrame(data)
# Feature_Extraction().convert_to_csv(data,'MFCCFeatures')
    
df=Feature_Extraction().extract_mfcc_from_folder('C:/Users/atulk/Downloads/20230522_031241500.wav')
print(df)