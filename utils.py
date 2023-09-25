# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 17:33:21 2023

@author: shangfr
"""
import json

def update_oss_data(emb_json):

    # Exporting to Embedding Projector FormatPermalink
    # output.tsv: This file should contain the embeddings without any headers.
    # metadata.tsv: This file should contain the original text and labels for the embeddings
    
    oss_json = json.load(open('./oss_data/oss_demo_projector_config.json'))
    oss_json['embeddings'] = [n for n in oss_json['embeddings']
                              if not n.get("tensorName") == emb_json["tensorName"]]
    oss_json['embeddings'].append(emb_json)
    with open('oss_data/oss_demo_projector_config.json', 'w+') as f:
        json.dump(oss_json, f, ensure_ascii=False, indent=4)




