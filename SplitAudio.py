from logging import error
from pydub import AudioSegment
from pydub.silence import split_on_silence
import glob
import os.path
import pandas as pd


# Split the wav files into chunks

# loop thru the folders
#folder_path = r"dataset\ReadText\HC\*.wav"
def split_into_chunks():
    folder_paths = ''
    

    for i in range(len(folder_paths)):
        for file in glob.glob(folder_paths[i]):
            try:
                print(file)
                #split to get HC\ID00
                path2, filename2 = os.path.split(file)
                root, ext = os.path.splitext(filename2)
                x = root.split('_')[0]

                directory = x
               #r"dataset\MDVR\HC"

                path = os.path.join(directory)
                print(path)
                os.makedirs(path)

                filename = file
                sound_file = AudioSegment.from_wav(filename)
                audio_chunks = split_on_silence(sound_file, 
                    # must be silent for at least half a second
                    min_silence_len=1000,
                    # consider it silent if quieter than -16 dBFS
                    silence_thresh=-40)
                for j, chunk in enumerate(audio_chunks):
                    out_file = path + "/chunk{0}.wav".format(j)
                    print ("exporting", out_file)
                    chunk.export(out_file, format="wav")
            except Exception as e:
                print(e)
                print("error while handling file: ", file)






   