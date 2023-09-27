# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 17:30:45 2023

@author: shangfr
"""
from optimum.onnxruntime import ORTModelForFeatureExtraction
from transformers import AutoTokenizer

model_checkpoint="sentence-transformers/all-MiniLM-L6-v2"
save_directory = "model_files/"

# Load a model from transformers and export it to ONNX
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
ort_model = ORTModelForFeatureExtraction.from_pretrained(model_checkpoint, export=True)

# Save the ONNX model and tokenizer
ort_model.save_pretrained(save_directory)
tokenizer.save_pretrained(save_directory)

# 模型量化
from optimum.onnxruntime.configuration import AutoQuantizationConfig
from optimum.onnxruntime import ORTQuantizer

# Define the quantization methodology
qconfig = AutoQuantizationConfig.arm64(is_static=False, per_channel=False)
quantizer = ORTQuantizer.from_pretrained(ort_model)

# Apply dynamic quantization on the model
quantizer.quantize(save_dir=save_directory, quantization_config=qconfig)

