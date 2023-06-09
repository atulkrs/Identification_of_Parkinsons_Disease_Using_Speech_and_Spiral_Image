import pandas as pd
df=pd.read_csv('AudioFeatureData.csv')
df=df.sample(frac=1)
df.to_csv('Audiofeatures.csv')