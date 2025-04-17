from transformers import AutoModel, AutoTokenizer
from pathlib import Path

import torch

# Saving util 
def save_embedding(embed, save_path):
    path = Path(save_path)
    if path.exists():
        print("File name already exists")
    else:
        torch.save(embed, save_path)

# BLaIR embedding model on HuggingFace (roberta_base)
def BLaIR_roberta_base_text_embedding_model(description_list, batch_size = 64, max_length = 512, device = None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    tokenizer = AutoTokenizer.from_pretrained("hyp1231/blair-roberta-base")
    model = AutoModel.from_pretrained("hyp1231/blair-roberta-base")
    
    model.to(device) # move to gpu
    embeddings_list = []

    for i in range(0, len(description_list), batch_size):
        batch = description_list[i:i + batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True, max_length = max_length, return_tensors="pt")
        for key in inputs.keys():
            inputs[key] = inputs[key].to(device)

        with torch.no_grad():
            batch_embeddings = model(**inputs, return_dict=True).last_hidden_state[:, 0]
            batch_embeddings = batch_embeddings / batch_embeddings.norm(dim=1, keepdim=True)

        embeddings_list.append(batch_embeddings.cpu()) 

    embeddings = torch.cat(embeddings_list, dim=0)
    return embeddings

# Trained blair embeeding model on the beauty data 
def custom_BLaIR_text_embedding_model(description_list, model_dir, batch_size = 64, max_length = 512, device = None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModel.from_pretrained(model_dir)
    
    model.to(device) # move to gpu
    embeddings_list = []

    for i in range(0, len(description_list), batch_size):
        batch = description_list[i:i + batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True, max_length = max_length, return_tensors="pt")
        for key in inputs.keys():
            inputs[key] = inputs[key].to(device)

        with torch.no_grad():
            batch_outputs = model(**inputs)
            batch_embeddings = batch_outputs.last_hidden_state[:, 0, :]

        embeddings_list.append(batch_embeddings.cpu()) 

    embeddings = torch.cat(embeddings_list, dim=0)
    return embeddings