import os
import pickle

import numpy as np
import pandas as pd
from datasets import load_dataset
import torch
from sklearn.manifold import TSNE

from embeddings.constants import CLEANED_DATA_PATH, BLAIR_MODEL_PATH
from utils.graph_helpers import train_model, plot_loss, final_evaluation, plot_embedding_features
from utils.graph_model import BaseGNNRecommender
from utils.setup_embeddings import e5_embedding_model, custom_BLaIR_text_embedding_model, \
    BLaIR_roberta_base_text_embedding_model


class EmbeddingAndGNNWrapper:
    def __init__(self, embedding_model_name="intfloat/e5-small-v2", pooling="mean", batch_size=64, max_length=512,
                 device=None):
        self.embedding_model_name = embedding_model_name
        self.pooling = pooling
        self.batch_size = batch_size
        self.max_length = max_length
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # concat'd features
        self.user_features = None
        self.product_features = None

        self.train_edges = pd.read_parquet(f"{CLEANED_DATA_PATH}/train_edges.parquet")
        self.test_edges = pd.read_parquet(f"{CLEANED_DATA_PATH}/test_edges.parquet")
        self.val_edges = pd.read_parquet(f"{CLEANED_DATA_PATH}/val_edges.parquet")

        self.user_id_to_idx = pickle.load(open(f"{CLEANED_DATA_PATH}/user_id_to_idx.pkl", "rb"))
        self.prod_id_to_idx = pickle.load(open(f"{CLEANED_DATA_PATH}/prod_id_to_idx.pkl", "rb"))

    def fit(self, X, y=None):
        (user_features_numeric_agg, user_features_string_agg,
         product_features_numeric, product_features_string,
         ) = X
        print('Embedding product meta and user reviews...')
        product_meta_embeddings, user_reviews_embeddings = self._embed(product_features_string,
                                                                       user_features_string_agg,
                                                                       self.embedding_model_name, self.pooling,
                                                                       self.max_length)

        self.product_features, self.user_features = self._prepare_combined_features(product_meta_embeddings,
                                                                                    user_reviews_embeddings,
                                                                                    product_features_numeric,
                                                                                    user_features_numeric_agg)
        print('Computation of embeddings complete!')
        num_users = len(self.user_id_to_idx)
        num_products = len(self.prod_id_to_idx)
        user_feature_dim = self.user_features.shape[1]
        product_feature_dim = self.product_features.shape[1]
        embedding_size = product_meta_embeddings.shape[1]

        train_edge_index = torch.tensor(self.train_edges[["user_idx", "prod_idx"]].to_numpy().T, dtype=torch.long)
        val_edge_index = torch.tensor(self.val_edges[["user_idx", "prod_idx"]].to_numpy().T, dtype=torch.long)
        self.test_edge_index = torch.tensor(self.test_edges[["user_idx", "prod_idx"]].to_numpy().T, dtype=torch.long)

        train_edge_weights = torch.tensor(self.train_edges.rating.to_list(), dtype=torch.float)
        val_edge_weights = torch.tensor(self.val_edges.rating.to_list(), dtype=torch.float)
        self.test_edge_weights = torch.tensor(self.test_edges.rating.to_list(), dtype=torch.float)

        train_edge_index = train_edge_index.to(self.device)
        train_edge_weights = train_edge_weights.to(self.device)
        val_edge_index = val_edge_index.to(self.device)
        val_edge_weights = val_edge_weights.to(self.device)
        self.user_features = self.user_features.to(self.device)
        self.product_features = self.product_features.to(self.device)

        self.base_gnn_model = BaseGNNRecommender(num_users, num_products, user_feature_dim, product_feature_dim,
                                                 embedding_size, custom_embedding=True).to(self.device)
        optimizer = torch.optim.Adam(self.base_gnn_model.parameters(), lr=0.01)

        print("Training model now...")

        train_loss, valid_loss, self.best_model = train_model(
            self.base_gnn_model,
            train_edge_index, train_edge_weights,
            val_edge_index, val_edge_weights,
            self.user_features, self.product_features,
            num_epochs=5,
            print_progress=True
        )

        train_df = pd.DataFrame({'train_loss': train_loss, 'valid_loss': valid_loss})
        train_df.to_csv(
            f'./{self.embedding_model_name}_{self.pooling}_{self.max_length}_train.csv')

        self._plot_tsne(product_meta_embeddings, user_reviews_embeddings)
        # use if passing in true y
        # self.y_true_ = y_true

        return self

    def score(self, X=None, y=None):
        rmse = final_evaluation(self.base_gnn_model, self.test_edge_index, self.test_edge_weights,
                                self.user_features, self.product_features, self.best_model)
        return rmse

    def _embed(self, product_features_string, user_features_string_agg,
               embedding_model_name, pooling, max_length):

        product_meta_features, user_review_features = None, None
        pdt_meta_emb_path = f"./{embedding_model_name}_pdt_meta_{pooling}_{max_length}.pt"

        if os.path.exists(pdt_meta_emb_path):
            product_meta_features = torch.load(pdt_meta_emb_path)
            user_review_features = torch.load(
                f"./{embedding_model_name}_user_rev_{pooling}_{max_length}.pt")
            product_meta_features = product_meta_features.to(self.device)
            user_review_features = user_review_features.to(self.device)
            return product_meta_features, user_review_features

        if embedding_model_name == 'E5':
            # TODO: meta col + 1/2 of other cols
            product_meta_features = e5_embedding_model(product_features_string["meta"], batch_size=64,
                                                       max_length=max_length, pooling=pooling)
            torch.save(product_meta_features, f"./E5_pdt_meta_{pooling}_{max_length}.pt")
            user_review_features = e5_embedding_model(user_features_string_agg["reviews"], batch_size=64,
                                                      max_length=max_length, pooling=pooling)
            torch.save(user_review_features, f"./E5_user_rev_{pooling}_{max_length}.pt")
        elif embedding_model_name == 'custom-blair':
            product_meta_features = custom_BLaIR_text_embedding_model(product_features_string["meta"],
                                                                      BLAIR_MODEL_PATH,
                                                                      batch_size=64, max_length=max_length,
                                                                      pooling=pooling)
            torch.save(product_meta_features, f"./custom-blair_pdt_meta_{pooling}_{max_length}.pt")
            user_review_features = custom_BLaIR_text_embedding_model(user_features_string_agg["reviews"],
                                                                     BLAIR_MODEL_PATH,
                                                                     batch_size=64, max_length=max_length,
                                                                     pooling=pooling)
            torch.save(user_review_features, f"./custom-blair_user_rev_{pooling}_{max_length}.pt")
        else:
            # use default blair
            product_meta_features = BLaIR_roberta_base_text_embedding_model(product_features_string["meta"],
                                                                            batch_size=64, max_length=max_length,
                                                                            pooling=pooling)
            torch.save(product_meta_features, f"./default-blair_pdt_meta_{pooling}_{max_length}.pt")
            user_review_features = BLaIR_roberta_base_text_embedding_model(user_features_string_agg["reviews"],
                                                                           batch_size=64, max_length=max_length,
                                                                           pooling=pooling)
            torch.save(user_review_features, f"./default-blair_user_rev_{pooling}_{max_length}.pt")

        product_meta_features = product_meta_features.to(self.device)
        user_review_features = user_review_features.to(self.device)
        print('hokay')
        return product_meta_features, user_review_features

    def _prepare_combined_features(self, product_meta_features, user_review_features, product_features_numeric,
                                   user_features_numeric_agg):
        product_features_numeric.main_category = product_features_numeric.main_category.apply(
            lambda x: 1 if x == "All Beauty" else 0)
        prod_feat_num = torch.tensor(product_features_numeric.drop(["parent_asin", "price"], axis=1).to_numpy(),
                                     dtype=torch.float)
        user_features_num = torch.tensor(user_features_numeric_agg.drop("user_id", axis=1).to_numpy(),
                                         dtype=torch.float)

        prod_feat_num = prod_feat_num.to(self.device)
        user_features_num = user_features_num.to(self.device)

        product_features = torch.cat([product_meta_features, prod_feat_num], dim=1)
        user_features = torch.cat([user_review_features, user_features_num], dim=1)

        return product_features, user_features

    def _plot_tsne(self, product_meta_embeddings, user_reviews_embeddings):
        review_features_numpy = user_reviews_embeddings.numpy()
        meta_features_numpy = product_meta_embeddings.numpy()

        tsne = TSNE(n_components=2, random_state=42)
        review_features_2d = tsne.fit_transform(review_features_numpy)
        meta_features_2d = tsne.fit_transform(meta_features_numpy)

        sample_indices = np.random.choice(review_features_2d.shape[0], size=5000, replace=False)
        review_features_2d_sampled = review_features_2d[sample_indices, :]
        meta_features_2d_sampled = meta_features_2d[sample_indices, :]

        plot_embedding_features(review_features_2d_sampled, save=True, name='review_tsne')
        plot_embedding_features(meta_features_2d_sampled, save=True, name='meta_tsne')

