import pickle

import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
import torch

from embeddings.constants import CLEANED_DATA_PATH
from utils.graph_helpers import train_model, plot_loss, final_evaluation
from utils.graph_model import BaseGNNRecommender
from utils.setup_embeddings import e5_embedding_model, custom_BLaIR_text_embedding_model, \
    BLaIR_roberta_base_text_embedding_model


# change the parent class later
class EmbeddingAndGNNWrapper:
    def __init__(self, embedding_model_name="intfloat/e5-small-v2", pooling="mean", batch_size=64, max_length=512, device=None):
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
        print(self.embedding_model_name)
        product_meta_embeddings, user_reviews_embeddings = self._embed(product_features_string, user_features_string_agg,
                                                                       self.embedding_model_name, self.pooling, self.max_length)

        self.product_features, self.user_features = self._prepare_combined_features(product_meta_embeddings, user_reviews_embeddings,
                                                                          product_features_numeric, user_features_numeric_agg)

        num_users = len(self.user_id_to_idx)
        num_products = len(self.prod_id_to_idx)
        user_feature_dim = self.user_features.shape[1]
        product_feature_dim = self.product_features.shape[1]
        embedding_size = product_meta_embeddings.shape[1]

        train_edge_index = torch.tensor(self.train_edges[["user_idx", "prod_idx"]].to_numpy().T, dtype=torch.long)[0:5]
        val_edge_index = torch.tensor(self.val_edges[["user_idx", "prod_idx"]].to_numpy().T, dtype=torch.long)[0:5]
        self.test_edge_index = torch.tensor(self.test_edges[["user_idx", "prod_idx"]].to_numpy().T, dtype=torch.long)[0:5]

        train_edge_weights = torch.tensor(self.train_edges.rating.to_list(), dtype=torch.float)[0:5]
        val_edge_weights = torch.tensor(self.val_edges.rating.to_list(), dtype=torch.float)[0:5]
        self.test_edge_weights = torch.tensor(self.test_edges.rating.to_list(), dtype=torch.float)[0:5]

        print(product_feature_dim, 'product feature dim')
        print(embedding_size, 'embedding size')
        print(train_edge_weights.shape, 'train edge weights dim')
        self.base_gnn_model = BaseGNNRecommender(num_users, num_products, user_feature_dim, product_feature_dim, embedding_size, custom_embedding=True).to(self.device)
        optimizer = torch.optim.Adam(self.base_gnn_model.parameters(), lr=0.01)

        train_loss, valid_loss, self.best_model = train_model(
            self.base_gnn_model,
            train_edge_index, train_edge_weights,
            val_edge_index, val_edge_weights,
            self.user_features, self.product_features,
            num_epochs=5,
            print_progress=False
        )

        plot_loss(train_loss, valid_loss)

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
        print('embedding model name in _embed', embedding_model_name)
        if embedding_model_name == 'E5':
            print('in e5')
            # TODO: meta col + 1/2 of other cols
            product_meta_features = e5_embedding_model(product_features_string["meta"], batch_size=64,
                                                       max_length=max_length, pooling=pooling)
            torch.save(product_meta_features, f"./e5_pdt_meta_{pooling}_{max_length}.pt")
            user_review_features = e5_embedding_model(user_features_string_agg["reviews"], batch_size=64,
                                                      max_length=max_length, pooling=pooling)
            torch.save(user_review_features, f"./e5_user_rev_{pooling}_{max_length}.pt")
        elif embedding_model_name == 'roberta-massive':
            product_meta_features = custom_BLaIR_text_embedding_model(product_features_string["meta"],
                                                                      "blair-roberta-base_massive",
                                                                      batch_size=64, max_length=max_length, pooling=pooling),
            torch.save(product_meta_features, f"./customblair_pdt_meta_{pooling}_{max_length}.pt")
            user_review_features = custom_BLaIR_text_embedding_model(user_features_string_agg["reviews"],
                                                                     "blair-roberta-base_massive",
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


#scikitlearn have gpu support now? can try, dont need only use cpu
