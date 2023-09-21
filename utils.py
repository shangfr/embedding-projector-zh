# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 17:33:21 2023

@author: shangfr
"""
import json
import pandas as pd
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')
print("Max Sequence Length:", model.max_seq_length)


def set_oss_data(emb_json):

    oss_json = json.load(open('./oss_data/oss_demo_projector_config.json'))
    oss_json['embeddings'] = [n for n in oss_json['embeddings']
                              if not n.get("tensorName") == emb_json["tensorName"]]
    oss_json['embeddings'].append(emb_json)
    with open('oss_data/oss_demo_projector_config.json', 'w+') as f:
        json.dump(oss_json, f, ensure_ascii=False, indent=4)


def get_embeddings(sentences):
    return model.encode(sentences, batch_size=32, show_progress_bar=True)


def df2emb(df, emb_col_name, name='Test', save=True, path="./oss_data/"):

    emb = get_embeddings(df[emb_col_name])
    # Exporting to Embedding Projector FormatPermalink
    # output.tsv: This file should contain the embeddings without any headers.
    # metadata.tsv: This file should contain the original text and labels for the embeddings
    # Convert NumPy array of embedding into data frame
    emb_df = pd.DataFrame(emb)

    if save:
        # Save dataframe as as TSV file without any index and header
        tensor_file = f'{path + name}_tensor.tsv'
        meta_file = f'{path + name}_meta.tsv'
        emb_df.to_csv(tensor_file, sep='\t', index=False, header=None)
        # Save dataframe without any index
        df.to_csv(meta_file, sep='\t', index=False)
        tensor_shape = [emb_df.shape[1], emb_df.shape[0]]
        emb_json = {"tensorName": name,
                    "tensorShape": tensor_shape,
                    "tensorPath": tensor_file,
                    "metadataPath": meta_file}
        set_oss_data(emb_json)

    return emb_df
