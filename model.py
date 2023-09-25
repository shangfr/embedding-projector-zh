# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 14:14:34 2023

@author: shangfr
"""
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')
print("Max Sequence Length:", model.max_seq_length)

def get_embeddings(sentences):
    '''
    Parameters
    ----------
    sentences : [str]
        输入语句文本.

    Returns
    -------
    embeddings : [float]
        输出embeddings向量.

    '''
    embeddings = model.encode(sentences, batch_size=32, show_progress_bar=True)
    return embeddings

