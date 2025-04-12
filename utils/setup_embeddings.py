from transformers import AutoModel, AutoTokenizer

import torch


# BLaIR embedding model on HuggingFace (roberta_base)
def BLaIR_roberta_base_text_embedding_model(description_list, batch_size=64, max_length=512, device=None,
                                            pooling='cls'):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # description_list = list(description_list.apply(lambda x: ' '.join(x))) # consider to drop
    tokenizer = AutoTokenizer.from_pretrained("hyp1231/blair-roberta-base")
    model = AutoModel.from_pretrained("hyp1231/blair-roberta-base")

    model.to(device)  # move to gpu
    embeddings_list = []

    for i in range(0, len(description_list), batch_size):
        batch = description_list[i:i + batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        for key in inputs.keys():
            inputs[key] = inputs[key].to(device)

        with torch.no_grad():
            last_hidden_state = model(**inputs, return_dict=True).last_hidden_state
            attention_mask = inputs["attention_mask"]
            # resize mask to hidden state size
            mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size())

            if pooling == "cls":
                batch_embeddings = last_hidden_state[:, 0]
            if pooling == "mean":
                sum_hidden = torch.sum(last_hidden_state * mask_expanded, 1)
                sum_mask = mask_expanded.sum(1)
                batch_embeddings = sum_hidden / sum_mask.clamp(min=1e-9)
            if pooling == "max":
                last_hidden_state[mask_expanded == 0] = float('-inf')
                batch_embeddings, _ = last_hidden_state.max(dim=1)

            # normalize
            batch_embeddings = batch_embeddings / batch_embeddings.norm(dim=1, keepdim=True)

        embeddings_list.append(batch_embeddings.cpu())

    embeddings = torch.cat(embeddings_list, dim=0)
    return embeddings


# custom blair embedding model
# difference in the way outputs are taken? 
def custom_BLaIR_text_embedding_model(description_list, model_dir, batch_size=64, max_length=512, device=None,
                                      pooling='cls'):

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # description_list = list(description_list.apply(lambda x: ' '.join(x))) # consider to drop
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModel.from_pretrained(model_dir)

    model.to(device)  # move to gpu
    embeddings_list = []

    for i in range(0, len(description_list), batch_size):
        batch = description_list[i:i + batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        for key in inputs.keys():
            inputs[key] = inputs[key].to(device)

        with torch.no_grad():
            last_hidden_state = model(**inputs).last_hidden_state
            attention_mask = inputs["attention_mask"]
            mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size())

            if pooling == "cls":
                batch_embeddings = last_hidden_state[:, 0, :]
            if pooling == "mean":
                sum_hidden = torch.sum(last_hidden_state * mask_expanded, 1)
                sum_mask = mask_expanded.sum(1)
                batch_embeddings = sum_hidden / sum_mask.clamp(min=1e-9)
            if pooling == "max":
                last_hidden_state[mask_expanded == 0] = float('-inf')
                batch_embeddings, _ = last_hidden_state.max(dim=1)

        embeddings_list.append(batch_embeddings.cpu())

    embeddings = torch.cat(embeddings_list, dim=0)
    return embeddings


def e5_embedding_model(description_list, batch_size=64, max_length=512, device=None, pooling="cls"):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # description_list = list(description_list.apply(lambda x: ' '.join(x))) # consider to drop
    tokenizer = AutoTokenizer.from_pretrained("intfloat/e5-small-v2")
    model = AutoModel.from_pretrained("intfloat/e5-small-v2")

    model.to(device)  # move to gpu
    embeddings_list = []
    description_list = ("passage: " + description_list.astype(str)).tolist()

    for i in range(0, len(description_list), batch_size):
        batch = description_list[i:i + batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        for key in inputs.keys():
            inputs[key] = inputs[key].to(device)

        with torch.no_grad():
            last_hidden_state = model(**inputs).last_hidden_state
            attention_mask = inputs["attention_mask"]
            mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size())

            if pooling == "cls":
                batch_embeddings = last_hidden_state[:, 0, :]
            if pooling == "mean":
                sum_hidden = torch.sum(last_hidden_state * mask_expanded, 1)
                sum_mask = mask_expanded.sum(1)
                batch_embeddings = sum_hidden / sum_mask.clamp(min=1e-9)
            if pooling == "max":
                last_hidden_state[mask_expanded == 0] = float('-inf')
                batch_embeddings, _ = last_hidden_state.max(dim=1)

        embeddings_list.append(batch_embeddings.cpu())

    embeddings = torch.cat(embeddings_list, dim=0)

    return embeddings


def instantiate_users(arg_1, arg_2=None):
    # Either pass in two number for the user nodes (num user, num features) 
    # Or pass in a df of the user_id to the required features 
    if type(arg_1) == int:
        return torch.zeros((arg_1, arg_2), dtype=torch.float)
    t1 = torch.tensor(arg_1.drop(["user_id", "reviews"], axis=1).to_numpy(), dtype=torch.float)
    return torch.cat([t1, arg_2], dim=1)
