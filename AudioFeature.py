import librosa
import numpy as np
import pandas as pd
import os

def zcr(data,frame_length=2048,hop_length=512):
    try:
        zcr=librosa.feature.zero_crossing_rate(data,frame_length=frame_length,hop_length=hop_length)
        return np.squeeze(zcr)
    except:
        pass
def rms(data,frame_length=2048,hop_length=512):
    rms=librosa.feature.rms(y=data,frame_length=frame_length,hop_length=hop_length)
    return np.squeeze(rms)
def mfcc(data,sr,frame_length=2048,hop_length=512,flatten:bool=True):
    mfcc=librosa.feature.mfcc(y=data,sr=sr)
    return np.squeeze(mfcc.T)if not flatten else np.ravel(mfcc.T) 
  

def extract_features(data,sr,frame_length,hop_length):
    result=np.array([])
    
    result=np.hstack((result,
                      zcr(data,frame_length,hop_length),
                      rms(data,frame_length,hop_length),
                      mfcc(data,sr,frame_length,hop_length),
                     
                     ))
    # result = result.reshape(1,-1)
    return result
# data,sr=librosa.load('C:/Users/atulk/Downloads/20230524_004034043.wav',duration=2.5,offset=0.6)
    
# aud=extract_features(data,sr,2048,512)
# print(aud.shape)
# data=[]
# path='Audio File/'
# for folder in os.listdir(path):
#     if(folder=='HC'):
#         fullpath=os.path.join(path,folder)
#         for file in os.listdir(fullpath):
#             audio,sr=librosa.load(os.path.join(path,folder,file),duration=2.5,offset=0.6)
#             temp=extract_features(audio,sr,2048,512)
#             temp.append(0)
#             data.append(temp)
#     else:
#         fullpath=os.path.join(path,folder)
#         for file in os.listdir(fullpath):
#             audio,sr=librosa.load(os.path.join(path,folder,file),duration=2.5,offset=0.6)
#             temp=extract_features(audio,sr,2048,512)
#             temp.append(1)
#             data.append(temp)

# df=pd.DataFrame(data)
# df.to_csv('UpdatedAudioFeature.csv')
# print('Done')
        
