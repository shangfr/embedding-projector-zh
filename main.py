# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 14:53:35 2023

@author: shangfr
"""

import pandas as pd
from utils import df2emb

sentences = ['This framework generates embeddings for each input sentence',
             'Sentences are passed as a list of string.',
             'The quick brown fox jumps over the lazy dog.']
label = ['yes', 'no', 'yes']

df = pd.DataFrame({"sentences": sentences, "label": label})
emb_col_name = "sentences"

emb_df = df2emb(df, emb_col_name, name='Test', save=True, path="oss_data/")
