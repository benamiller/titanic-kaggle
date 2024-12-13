import pandas as pd
import numpy as np

print(np.zeros((1,2)))

df = pd.read_csv('./data/train.csv')
print(df.head(5))

# Check for nullish Sex values
print(df[df['Sex'].isna()])
