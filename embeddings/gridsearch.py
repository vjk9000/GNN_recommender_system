import os

import pandas as pd
import torch
from torch.nn.functional import normalize
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, AutoConfig

from sklearn.model_selection import GridSearchCV

from embedding_gs_wrapper import EmbeddingAndGNNWrapper


def gs_embeddings():
    user_features_numeric_agg = pd.read_parquet("cleaned_v2/user_features_numeric_agg.parquet")
    user_features_string_agg = pd.read_parquet("cleaned_v2/user_features_string_agg.parquet")
    product_features_numeric = pd.read_parquet("cleaned_v2/product_features_numeric.parquet")
    product_features_string = pd.read_parquet("cleaned_v2/product_features_string.parquet")

    train_edges = pd.read_parquet("cleaned_v2/train_edges.parquet")
    test_edges = pd.read_parquet("cleaned_v2/test_edges.parquet")
    val_edges = pd.read_parquet("cleaned_v2/val_edges.parquet")

    X_custom = (
        user_features_numeric_agg,
        user_features_string_agg,
        product_features_numeric,
        product_features_string,
        train_edges,
        test_edges,
        val_edges)

    # will be passed into init of wrapper
    param_grid = {
        "pooling": ["cls", "max"],
        "max_length": [2],
        "embedding_model_name": "E5"
    }

    grid = GridSearchCV(
        EmbeddingAndGNNWrapper(),
        param_grid=param_grid,
        scoring=None,  # uses "score" method in class by default
        cv=[(slice(None), slice(None))],  # dummy CV, since train/val is inside
        verbose=3
    )

    grid.fit([X_custom])


if __name__ == "__main__":
  gs_embeddings()