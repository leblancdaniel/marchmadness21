import pandas as pd 
import numpy as np 
import os 


current_dir = os.path.dirname(__file__)
data_dir = os.path.join(current_dir, "data")
data = {}
# LOAD CSV 
for f in os.listdir(data_dir):
    if f.endswith(".csv"):
        fname = f[:-4].lower()
        df_name = fname + '_df'
        data[df_name] = pd.read_csv(os.path.join(data_dir, f))
    else:
        continue
# List of dfs, for reference
df_ls = []
for df in data:
    df_ls.append(df)
print(df_ls)
# To access a DF in this dictionary of DF use format: data["df_name"]
print(data)