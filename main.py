# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 14:53:35 2023

@author: shangfr
"""
import pandas as pd
from utils import update_oss_data
from model import get_embeddings

df = pd.read_parquet("data/project_brief.parquet")
dfa = df.sample(n=1000).reset_index(drop=True)

emb_col_name = "brief"
emb = get_embeddings(dfa[emb_col_name])
# Convert NumPy array of embedding into data frame
emb_df = pd.DataFrame(emb)
# Save dataframe as as TSV file without any index and header
# Save dataframe without any index
name = 'Test'
path = "oss_data/"

meta_file = f'{path + name}_meta.tsv'
tensor_file = f'{path + name}_tensor.tsv'

df.to_csv(meta_file, sep='\t', index=False)
emb_df.to_csv(tensor_file, sep='\t', index=False, header=None)

tensor_shape = [emb_df.shape[1], emb_df.shape[0]]
emb_json = {"tensorName": name,
            "tensorShape": tensor_shape,
            "tensorPath": tensor_file,
            "metadataPath": meta_file}
update_oss_data(emb_json)
