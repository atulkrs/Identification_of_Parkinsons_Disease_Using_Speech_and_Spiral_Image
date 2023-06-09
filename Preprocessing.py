from pydub import AudioSegment
from pydub.silence import split_on_silence
import os

# Set the directory containing the audio files
audio_directory = 'Audio File/HP1/'

# Define the noise reduction function
def reduce_noise(audio_clip):
    # Set the noise threshold to -50 dBFS
    noise_threshold = -50.0
    print('starting noise reduction')
    # Split the audio clip into segments on silence
    segments = split_on_silence(audio_clip, min_silence_len=1000, silence_thresh=noise_threshold)
    # Concatenate the non-silent segments into a new audio clip
    reduced_clip = segments[0]
    for segment in segments[1:]:
        reduced_clip += segment
    # Return the reduced clip
    return reduced_clip

# Define the normalization function
def normalize_audio(audio_clip):
    # Set the target peak amplitude to -3 dBFS
    target_amplitude = -3.0
    print('starting normalization')
    # Normalize the audio clip to the target amplitude
    normalized_clip = audio_clip.normalize(headroom=target_amplitude)
    # Return the normalized clip
    return normalized_clip

# Loop over all files in the directory
for file_name in os.listdir(audio_directory):
    # Check if the file is a WAV file
    if file_name.endswith('.wav'):
        # Load the audio file
        file_path = os.path.join(audio_directory, file_name)
        audio_clip = AudioSegment.from_wav(file_path)
        print(file_path)
        # Apply noise reduction
        reduced_clip = reduce_noise(audio_clip)

        # Apply normalization
        normalized_clip = normalize_audio(reduced_clip)

        # Save the preprocessed audio file
        preprocessed_file_name = 'preprocessed_' + file_name
        preprocessed_file_path = os.path.join(audio_directory, preprocessed_file_name)
        normalized_clip.export(preprocessed_file_path, format='wav')
