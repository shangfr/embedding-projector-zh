# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 14:53:35 2023

@author: shangfr
"""
import pandas as pd
from utils import update_oss_data
from model import EmbPipeline
emb_model = EmbPipeline()

df = pd.read_parquet("data/ipc.parquet")
df.fillna("",inplace=True)
emb_col_name = "描述"

df = df.loc[(df['绿色国际'] !="") & (df[emb_col_name]!="")]


emb = emb_model(df[emb_col_name].tolist())

emb_df = pd.DataFrame(emb)

# Save dataframe as as TSV file without any index and header
# Save dataframe without any index
name = 'GREEN_IPC_IPC'
path = "oss_data/"

meta_file = f'{path + name}_meta.tsv'
tensor_file = f'{path + name}_tensor.tsv'
#emb_df = pd.read_csv(tensor_file, sep='\t', header=None)

del df[emb_col_name]

df.to_csv(meta_file, sep='\t', index=False)
emb_df.to_csv(tensor_file, sep='\t', index=False, header=None)

tensor_shape = [emb_df.shape[1], emb_df.shape[0]]
emb_json = {"tensorName": name,
            "tensorShape": tensor_shape,
            "tensorPath": tensor_file,
            "metadataPath": meta_file}
update_oss_data(emb_json)
