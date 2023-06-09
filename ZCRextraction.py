import librosa
import numpy as np
import csv
import os

# Load the audio file
audio_dir ='Audio File\HP'

for audio_file in os.listdir(audio_dir):
    if audio_file.endswith(".wav"):
        audio_file_path = os.path.join(audio_dir, audio_file)
        
        # Load the audio file
        y, sr = librosa.load(audio_file_path)

# Calculate the zero crossing rate
zcr = librosa.feature.zero_crossing_rate(y)
print('hi')
# Calculate other related features
f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
print('hi')
f0_mean = np.mean(f0[f0 > 0])
print('hi')
f0_std_deviation = np.std(f0[f0 > 0])
print('hi')
harmonicity = librosa.effects.harmonic(y)
print('hi')

print("Zero crossing rate:", np.mean(zcr))
print("F0 mean:", f0_mean)
print("F0 standard deviation:", f0_std_deviation)
print("Harmonicity:", np.mean(harmonicity))


# Save the extracted features to a CSV file
with open('ZCR.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Zero crossing rate', 'F0 mean', 'F0 standard deviation', 'Harmonicity'])
    writer.writerow([np.mean(zcr), f0_mean, f0_std_deviation, np.mean(harmonicity)])
