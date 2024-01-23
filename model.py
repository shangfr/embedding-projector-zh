# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 14:14:34 2023

@author: shangfr
"""
from optimum.onnxruntime import ORTModelForFeatureExtraction
from transformers import AutoTokenizer
from transformers import Pipeline
import torch.nn.functional as F
import torch


# copied from the model card
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


class EmbPipeline(Pipeline):
    def __init__(self,model_directory,model_name = "model.onnx"):
        model = ORTModelForFeatureExtraction.from_pretrained(model_directory, file_name=model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_directory)
        print("Loaded parameters from model.onnx")
        super(EmbPipeline, self).__init__(model,tokenizer)
        
    def _sanitize_parameters(self, **kwargs):
        # we don't have any hyperameters to sanitize
        preprocess_kwargs = {}
        return preprocess_kwargs, {}, {}

    def preprocess(self, inputs):
        encoded_inputs = self.tokenizer(inputs, padding=True, truncation=True, return_tensors='pt')
        return encoded_inputs

    def _forward(self, model_inputs):
        outputs = self.model(**model_inputs)
        return {"outputs": outputs, "attention_mask": model_inputs["attention_mask"]}

    def postprocess(self, model_outputs):
        # Perform pooling
        sentence_embeddings = mean_pooling(model_outputs["outputs"], model_outputs['attention_mask'])
        # Normalize embeddings
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        return sentence_embeddings.tolist()[0]


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
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print("Max Sequence Length:", model.max_seq_length)

    embeddings = model.encode(sentences, batch_size=32, show_progress_bar=True)
    return embeddings
