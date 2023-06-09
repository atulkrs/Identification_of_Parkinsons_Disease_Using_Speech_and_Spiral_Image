import glob
import pandas as pd
import librosa
import numpy as np

def extract_features(file):
    # Load audio file
    y, sr = librosa.load(file)
     
    # Extract RMS feature
    rms = librosa.feature.rms(y=y)
    # Extract pitch
    pitch, _ = librosa.piptrack(y=y, sr=sr)
    pitch_mean = np.nanmean(pitch)
    pitch_std = np.nanstd(pitch)
    
    
    # Extract harmonicity
    harm = librosa.effects.harmonic(y=y)  
    harm_mean = np.mean(harm)
    # Extract HNR
    hnr = librosa.effects.harmonic(y=y, margin=1)
    hnr_mean = np.mean(hnr)
   
    # Extract point process
    pp = librosa.effects.harmonic(y=y, margin=1.0)
    pp_mean = np.mean(pp)
    
      
    # Create a feature list with all the features
    features = [rms[0][0], pitch_mean, pitch_std, harm_mean, hnr_mean, pp_mean,] 
    return features

# Define a list to store the features and labels
all_features = []


# Loop through all the audio files and extract features
for file in glob.glob('Audio File/ReadText/PD/*.wav'):
    print(file)
    features = extract_features(file)
    all_features.append(features)
    print(all_features)
    label = file.split('/')[-1].split('_')[0]
    

# Create a pandas dataframe with the features and labels
df = pd.DataFrame(data=all_features, columns=['RMS', 'Pitch_mean', 'Pitch_std', 'Harmonicity_mean', 'HNR_mean', 'PP_mean'])
df.to_csv('RMSFfeaturePD.csv')
