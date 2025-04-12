import os

import pandas as pd
import torch
from torch.nn.functional import normalize
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, AutoConfig

from utils.embedding_utils import add_sos_and_bos, aggregate_embeddings

E5 = "intfloat/e5-small-v2"


class GridSearch():
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps")
        # self.agg_method = 'mean'
        self.tokenizer = None
        self.model = None
        # self.model.eval()

    def embed(self):
        meta_features = torch.load("meta_features_512.pt")
        review_features = torch.load("review_features_512.pt")
        user_review_features = torch.load("./user_review_features_512.pt")

    def embed_text(self, hf_model_name, texts, batch_size=32):
        local_model_name = hf_model_name.replace("/", "_")
        # print(os.path.exists(f"./{local_model_name}.pt"))
        if os.path.exists(f"./{local_model_name}.pt"):
            return pd.read_csv(f"./{local_model_name}.csv").meta
        self.tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
        self.model = AutoModel.from_pretrained(hf_model_name).to(self.device)

        embeddings = []

        with torch.no_grad():
            for i in tqdm(range(0, len(texts), batch_size)):
                batch = texts[i:i + batch_size].tolist()
                inputs = self.tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(self.device)
                outputs = self.model(**inputs)
                emb = normalize(outputs.last_hidden_state[:, 0], p=2, dim=1)
                embeddings.append(emb)

        return torch.cat(embeddings, dim=0)

    def get_embeddings(self, model_name, product_df, user_df):
        if model_name == E5:
            product_df["formatted"] = "passage: " + product_df['meta'].astype(str)
            # adjust later to correct col
            # user_df["formatted"] = "passage: " + user_df['meta'].astype(str)
            product_df["embedding"] = list(gs.embed_text(model_name, product_df["formatted"]))
            # user_df["embedding"] = list(gs.embed_text(model_name, user_df["formatted"])

        return product_df, user_df


if __name__ == "__main__":
    # models = [BLAIR_BASE, BLAIR_CUSTOM, E5]
    # for model in models...
    torch.cuda.empty_cache()
    gs = GridSearch()
    print(os.getcwd())
    product_string_df = pd.read_parquet("../data/cleaned_v2/product_features_string.parquet")
    user_string_df = pd.read_parquet("../data/cleaned_v2/user_features_string_agg.parquet")

    product_string_df, user_string_df = gs.get_embeddings(E5, product_string_df, user_string_df)
    # just use meta col first
    # for e5 use formatted
    print(product_string_df.embedding)
