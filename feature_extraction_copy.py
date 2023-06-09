import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#import librosa 
#import librosa.display
#import IPython.display as ipd
# import Preprocessing as StandardScaler
import parselmouth
#import seaborn as sns
# from parselmouth.praat import call
import glob
import os.path
from datetime import datetime

class Feature_Extraction:
    """
    Feature extraction class containing the methods to extract features for each voice sample
    
    Attributes:
    
    mfcc: list
         a list of mfcc extracted from the voice sample 
            
    """
    
    def __init__(self):
        self.mfcc = []

   
    def extract_mfcc(self, voice_sample):

        sound = parselmouth.Sound(voice_sample)
        mfcc_object = sound.to_mfcc(number_of_coefficients=12) #the optimal number of coeefficient used is 12
        mfcc = mfcc_object.to_array()
        mfcc_mean = np.mean(mfcc.T,axis=0)
        return mfcc_mean

    def extract_mfcc_from_folder(self, folder_path):
        file_list =[]
        mfcc_list = []
        features = []
        curr_time = datetime.now()
        print("Entering extract_mfcc_from_folder, time:", curr_time)
        for file in glob.glob(folder_path):
            try:
                #print("Processing file:", file)
                mfcc_per_file = self.extract_mfcc(file)
                #mfcc_list.append(mfcc_for_file)
                #file_list.append(file)
                features.append([file, mfcc_per_file])
            except:
                print("error while handling file: ", file)
        #df = pd.DataFrame(file_list, mfcc_list)
        df = pd.DataFrame(features, columns=['voiceID','mfcc'])
        df[['mfcc_feature0','mfcc_feature1','mfcc_feature2', 'mfcc_feature3','mfcc_feature4','mfcc_feature5', 'mfcc_feature6', 'mfcc_feature7','mfcc_feature8', 'mfcc_feature9', 'mfcc_feature10','mfcc_feature11', 'mfcc_feature12']] = pd.DataFrame(df.mfcc.to_list())
        df = df.drop(columns=['mfcc'])
        return df


    def extract_features_from_folder(self, folder_path):
        file_list = []
        mean_F0_list = []
        sd_F0_list = []
        hnr_list = []
        localJitter_list = []
        localabsoluteJitter_list = []
        rapJitter_list = []
        ppq5Jitter_list = []
        localShimmer_list = []
        localdbShimmer_list = []
        apq3Shimmer_list = []
        aqpq5Shimmer_list = []
        curr_time = datetime.now()
        print("Entering extract_features_from_folder, time:", curr_time)
        for file in glob.glob(folder_path):
            #print("extract_features_from_folder: ", file)
            try:
                (meanF0, stdevF0, hnr, localJitter, localabsoluteJitter, rapJitter, ppq5Jitter, localShimmer, localdbShimmer, apq3Shimmer, aqpq5Shimmer) = self.extract_acoustic_features(file, 75, 500, "Hertz") 
                file_list.append(file) # make an ID list
                mean_F0_list.append(meanF0) # make a mean F0 list
                sd_F0_list.append(stdevF0) # make a sd F0 list
                hnr_list.append(hnr)
                localJitter_list.append(localJitter)
                localabsoluteJitter_list.append(localabsoluteJitter)
                rapJitter_list.append(rapJitter)
                ppq5Jitter_list.append(ppq5Jitter)
                localShimmer_list.append(localShimmer)
                localdbShimmer_list.append(localdbShimmer)
                apq3Shimmer_list.append(apq3Shimmer)
                aqpq5Shimmer_list.append(aqpq5Shimmer)
            except:
                print("missed:", file)
        df = pd.DataFrame(np.column_stack([file_list, mean_F0_list, sd_F0_list, hnr_list, localJitter_list, localabsoluteJitter_list, rapJitter_list, ppq5Jitter_list, localShimmer_list, localdbShimmer_list, apq3Shimmer_list, aqpq5Shimmer_list]), columns=['voiceID','meanF0Hz', 'stdevF0Hz', 'HNR', 'localJitter', 'localabsoluteJitter', 'rapJitter', 'ppq5Jitter', 'localShimmer', 'localdbShimmer', 'apq3Shimmer', 'apq5Shimmer'])  
        return df

   

    def convert_to_csv(self, df, filename):
        df.to_csv(filename+".csv", index=False)

        



