from torch_geometric.nn import GCNConv, SAGEConv

import torch
import torch.nn as nn
import torch.nn.functional as F

class Base_GNN_Model(nn.Module):
    def __init__(self, num_users, num_products, user_feature_dim, product_feature_dim, embedding_dim=64):
        super(Base_GNN_Model, self).__init__()

        # offset for prod
        self.offset = num_users

        # user and product embedding layers
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.product_embedding = nn.Embedding(num_products, embedding_dim)

        # feature transformation layers
        self.user_feature_transform = nn.Linear(user_feature_dim, embedding_dim)
        self.product_feature_transform = nn.Linear(product_feature_dim, embedding_dim)

        # GNN layers
        self.conv1 = GCNConv(embedding_dim, embedding_dim)
        self.conv2 = GCNConv(embedding_dim, embedding_dim)

        # prediction layer
        self.predictor = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 1)
        )

    def forward(self, edge_index, user_features, product_features):
        user_indices = edge_index[0]
        product_indices = edge_index[1]
        offset_product_embeddings = product_indices - self.offset

        # Make the combined edges
        forward_edge_index = torch.stack([user_indices, product_indices], dim=0)
        reverse_edge_index = torch.stack([product_indices, user_indices], dim=0)
        combined_edge_index = torch.cat([forward_edge_index, reverse_edge_index], dim=1)

        # transform features
        user_x = self.user_feature_transform(user_features) + self.user_embedding.weight
        product_x = self.product_feature_transform(product_features) + self.product_embedding.weight

        # combine user and product features
        x = torch.cat([user_x, product_x], dim=0)

        # # GNN layers
        x = F.relu(self.conv1(x, combined_edge_index))
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv2(x, combined_edge_index)

        # Split back the embeddings 
        user_embeddings = x[:len(user_x)]
        product_embeddings = x[len(user_x):]

        # embeddings for the specific user-product pairs in edge_index
        user_emb = user_embeddings[user_indices]
        product_emb = product_embeddings[offset_product_embeddings]

        # concat user and product embeddings for prediction
        pair_embeddings = torch.cat([user_emb, product_emb], dim=1)

        # predict ratings
        predictions = self.predictor(pair_embeddings).squeeze()

        # scale predictions 
        # predictions = torch.sigmoid(predictions) * 4 + 1

        return predictions