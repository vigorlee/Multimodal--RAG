import os
import torch
import pandas as pd
import numpy as np
from torchvision.transforms import ToPILImage
from transformers import AutoImageProcessor
os.environ["GIT_PYTHON_REFRESH"] = "quiet"
from multi_modal_lndex.FLMR.flmr import index_custom_collection, FLMRModelForIndexing
from multi_modal_lndex.FLMR.flmr import FLMRQueryEncoderTokenizer, FLMRContextEncoderTokenizer, FLMRModelForRetrieval

# load models
checkpoint_path = "D:\project\PreFLMR_ViT-B"
image_processor_name = "D:\project\clip-vit-base-patch16"

query_tokenizer = FLMRQueryEncoderTokenizer.from_pretrained(checkpoint_path, subfolder="query_tokenizer")
context_tokenizer = FLMRContextEncoderTokenizer.from_pretrained(
    checkpoint_path, subfolder="context_tokenizer"
)

model = FLMRModelForRetrieval.from_pretrained(
    checkpoint_path,
    query_tokenizer=query_tokenizer,
    context_tokenizer=context_tokenizer,
)
image_processor = AutoImageProcessor.from_pretrained(image_processor_name)
Q_pixel_values = torch.zeros(1, 3, 224, 224)

D_encoding = context_tokenizer(["a apple in","aaaaa","sssss"])
context_input_ids = D_encoding['input_ids']
context_attention_mask = D_encoding['attention_mask']

Q_encoding = query_tokenizer(["a apple in here"])
qcontext_input_ids = Q_encoding['input_ids']
qcontext_attention_mask = Q_encoding['attention_mask']

text_embeddings = model.doc(input_ids=context_input_ids, attention_mask=context_attention_mask)
D, D_mask = text_embeddings.late_interaction_output, text_embeddings.context_mask
Q_duplicated = model.query(input_ids=qcontext_input_ids, attention_mask=qcontext_attention_mask, concat_output_from_vision_encoder=False).late_interaction_output
Q_duplicated = Q_duplicated.repeat_interleave(3, dim=0).contiguous()
scores = model.score(Q_duplicated, D, D_mask)
print(scores)
