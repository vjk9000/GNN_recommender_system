import pickle

import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
import torch
import numpy as np

from utils.graph_helpers import train_model, plot_loss, final_evaluation
from utils.graph_model import BaseGNNRecommender
from utils.setup_embeddings import e5_embedding_model, custom_BLaIR_text_embedding_model, \
    BLaIR_roberta_base_text_embedding_model


# change the parent class later
class EmbeddingGSWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, embedding_model_name="intfloat/e5-small-v2", pooling="cls", batch_size=64, max_length=512, device=None):
        self.embedding_model_name = embedding_model_name
        self.pooling = pooling
        self.batch_size = batch_size
        self.max_length = max_length
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.user_features_numeric_agg = pd.read_parquet("cleaned_v2/user_features_numeric_agg.parquet")
        self.user_features_string_agg = pd.read_parquet("cleaned_v2/user_features_string_agg.parquet")
        self.product_features_numeric = pd.read_parquet("cleaned_v2/product_features_numeric.parquet")
        self.product_features_string = pd.read_parquet("cleaned_v2/product_features_string.parquet")
        self.train_edges = pd.read_parquet("cleaned_v2/train_edges.parquet")
        self.test_edges = pd.read_parquet("cleaned_v2/test_edges.parquet")
        self.val_edges = pd.read_parquet("cleaned_v2/val_edges.parquet")

        self.user_id_to_idx = pickle.load(open("cleaned_v2/user_id_to_idx.pkl", "rb"))
        self.prod_id_to_idx = pickle.load(open("cleaned_v2/prod_id_to_idx.pkl", "rb"))

        # TODO: see if need later
        self.embedding_size = 64

    def fit(self, X, y=None):

        # product string, user string, product num, user num, = X
        # Unpack inputs
        # (product_text_series, product_num_feats,
        #  user_num_feats,
        #  train_edge_index, train_edge_weights,
        #  val_edge_index, val_edge_weights,
        #  y_true, gnn_class, gnn_kwargs) = X
        #

        # TODO: pass in embedding model name, pooling, max length for GS

        product_meta_embeddings, user_reviews_embeddings = self._embed(self.product_features_string, self.user_features_string_agg,
                                                                       embedding_model_name, pooling, max_length)

        product_features, user_features = self._prepare_combined_features(product_meta_embeddings, user_reviews_embeddings,
                                                                          self.product_features_numeric, self.user_features_numeric_agg)

        num_users = len(self.user_id_to_idx)
        num_products = len(self.prod_id_to_idx)
        user_feature_dim = user_features.shape[1]
        product_feature_dim = product_features.shape[1]

        train_edge_index = torch.tensor(self.train_edges[["user_idx", "prod_idx"]].to_numpy().T, dtype=torch.long)
        val_edge_index = torch.tensor(self.val_edges[["user_idx", "prod_idx"]].to_numpy().T, dtype=torch.long)
        self.test_edge_index = torch.tensor(self.test_edges[["user_idx", "prod_idx"]].to_numpy().T, dtype=torch.long)

        train_edge_weights = torch.tensor(self.train_edges.rating.to_list(), dtype=torch.float)
        val_edge_weights = torch.tensor(self.val_edges.rating.to_list(), dtype=torch.float)
        self.test_edge_weights = torch.tensor(self.test_edges.rating.to_list(), dtype=torch.float)

        self.base_gnn_model = BaseGNNRecommender(num_users, num_products, user_feature_dim,product_feature_dim).to(self.device)
        optimizer = torch.optim.Adam(self.base_gnn_model.parameters(), lr=0.01)

        train_loss, valid_loss, self.best_model = train_model(
            self.base_gnn_model,
            train_edge_index, train_edge_weights,
            val_edge_index, val_edge_weights,
            user_features, product_features,
            num_epochs=5,
            print_progress=False
        )

        plot_loss(train_loss, valid_loss)

        # use if passing in true y
        # self.y_true_ = y_true
        return self

    def predict(self, X):
        # Your custom predict logic here
        raise NotImplementedError("Predict method not needed for GridSearch in this case")

    def score(self, X, y=None):
        # TODO: see how to return rmse for final_evaluation
        rmse = final_evaluation(self.base_gnn_model, self.test_edge_index, self.test_edge_weights,
                                self.user_features, self.product_features, self.best_model)
        return rmse

    def _embed(self, product_features_string, user_features_string_agg,
               embedding_model_name, pooling, max_length):
        product_meta_features, user_review_features = None, None

        if embedding_model_name == 'E5':
            product_meta_features = e5_embedding_model(product_features_string["meta"], batch_size=64,
                                                       max_length=max_length, pooling=pooling)
            torch.save(product_meta_features, f"./e5_pdt_meta_{pooling}_{max_length}.pt")
            user_review_features = e5_embedding_model(user_features_string_agg["reviews"], batch_size=64,
                                                      max_length=max_length, pooling=pooling)
            torch.save(user_review_features, f"./e5_user_rev_{pooling}_{max_length}.pt")

        if embedding_model_name == 'roberta-massive':
            product_meta_features = custom_BLaIR_text_embedding_model(product_features_string["meta"],
                                                                      "./embeddings/blair-roberta-base_massive",
                                                                      batch_size=64, max_length=max_length, pooling=pooling),
            torch.save(product_meta_features, f"./customblair_pdt_meta_{pooling}_{max_length}.pt")
            user_review_features = custom_BLaIR_text_embedding_model(user_features_string_agg["reviews"],
                                                      "./embeddings/blair-roberta-base_massive",
                                                      batch_size=64, max_length=max_length, pooling=pooling)
            torch.save(user_review_features, f"./customblair_user_rev_{pooling}_{max_length}.pt")

        else:
            # use default blair
            product_meta_features = BLaIR_roberta_base_text_embedding_model(product_features_string["meta"],
                                                                            batch_size=64, max_length=max_length, pooling=pooling),
            torch.save(product_meta_features, f"./blairbase_pdt_meta_{pooling}_{max_length}.pt")
            user_review_features = BLaIR_roberta_base_text_embedding_model(user_features_string_agg["reviews"], batch_size=64, max_length=max_length, pooling=pooling)
            torch.save(user_review_features, f"./blairbase_user_rev_{pooling}_{max_length}.pt")

        return product_meta_features, user_review_features

    def _prepare_combined_features(self, product_meta_features, user_review_features, product_features_numeric, user_features_numeric_agg):
        product_features_numeric.main_category = product_features_numeric.main_category.apply(
            lambda x: 1 if x == "All Beauty" else 0)
        prod_feat_num = torch.tensor(product_features_numeric.drop(["parent_asin", "price"], axis=1).to_numpy(),
                                     dtype=torch.float)
        product_features = torch.cat([product_meta_features, prod_feat_num], dim=1)

        user_features_num = torch.tensor(user_features_numeric_agg.drop("user_id", axis=1).to_numpy(),
                                         dtype=torch.float)
        user_features = torch.cat([user_review_features, user_features_num], dim=1)

        return product_features, user_features

