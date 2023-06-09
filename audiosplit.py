import os
import speech_recognition as sr
from pydub import AudioSegment
from gtts import gTTS


def get_vowels_from_audio(audio_file):
    recognizer = sr.Recognizer()

    with sr.AudioFile(audio_file) as source:
        audio = recognizer.record(source)  # Read the entire audio file

    # Use a speech recognition engine to convert audio to text
    text = recognizer.recognize_google(audio)

    vowels = [char for char in text if char.lower() in 'aeioubcdfghjklmnpqrstvwxyz']
    return vowels

def convert_text_to_audio(text, output_file):
    # Convert text to audio using pyttsx3
    tts = gTTS(text=text, lang='en')
    tts.save(output_file)

# Define the input and output folders
input_folder = 'Audio File\\ReadText\\HC'
output_folder = 'VowelsAudioforHC'

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Iterate over all audio files in the input folder
for filename in os.listdir(input_folder):
    if filename.endswith('.wav'):
        audio_file = os.path.join(input_folder, filename)
        vowels = get_vowels_from_audio(audio_file)

        file_name = os.path.splitext(filename)[0]


        # Iterate over vowels and save them as audio files in the output folder
        for i, vowel in enumerate(vowels):
            try:
                output_file = os.path.join(output_folder, f'{file_name}_{i+1}_conso.wav')
                convert_text_to_audio(vowel, output_file)
                print(f"Vowel {i+1} from {filename} saved as {output_file}")
            except Exception as e:
                print(f"Error processing vowel {i+1} from {filename}: {str(e)}")

