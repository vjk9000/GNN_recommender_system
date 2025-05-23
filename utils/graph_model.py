
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, GATv2Conv

import torch
import torch.nn as nn
import torch.nn.functional as F

class Base_GNN_Model(nn.Module):
    def __init__(self, num_users, num_products, user_feature_dim, product_feature_dim, embedding_dim=64, dropout_prob = 0.2):
        super(Base_GNN_Model, self).__init__()

        self.dropout_prob = dropout_prob

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
        x = F.dropout(x, p=self.dropout_prob, training=self.training)
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
    

class Experiment_1(nn.Module):
    def __init__(self, num_users, num_products, user_feature_dim, product_feature_dim, embedding_dim=64):
        super(Experiment_1, self).__init__()

        # feature transformation embedding_dim
        self.user_feature_transform = nn.Linear(user_feature_dim, embedding_dim)
        self.product_feature_transform = nn.Linear(product_feature_dim, embedding_dim)

        # GNN layers
        self.conv1 = GCNConv(embedding_dim, embedding_dim)
        self.conv2 = GCNConv(embedding_dim, embedding_dim)

        # prediction layer
        self.predictor = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 1)
        )

    def forward(self, edge_index, user_features, product_features):

        # Obtains raw user and product indices
        user_indices = edge_index[0]
        product_indices = edge_index[1] - user_features.shape[0]

        # transform features
        user_x = self.user_feature_transform(user_features)
        product_x = self.product_feature_transform(product_features)

        # combine user and product features
        x = torch.cat([user_x, product_x], dim=0)

        # combined edge index for message passing
        combined_edge_index = torch.cat([
            edge_index,
            torch.stack([edge_index[1], edge_index[0]], dim=0)
        ], dim=1)

        # GNN layers
        x = F.relu(self.conv1(x, combined_edge_index))
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv2(x, combined_edge_index)

        user_embeddings = x[:len(user_x)]
        product_embeddings = x[len(user_x):]

        # embeddings for the specific user-product pairs in edge_index
        user_emb = user_embeddings[user_indices]
        product_emb = product_embeddings[product_indices]

        # Normalize the embeddings
        user_emb_norm = F.normalize(user_emb, p=2, dim=-1)
        product_emb_norm = F.normalize(product_emb, p=2, dim=-1)
        
        # Compute element-wise dot product
        dot_product = user_emb_norm * product_emb_norm

        # predict ratings
        predictions = self.predictor(dot_product).squeeze()

        return predictions
    
class GNNRecommenderwithSkipConnections(nn.Module):
    def __init__(self, num_users, num_products, user_feature_dim, product_feature_dim, embedding_dim=64, dropout_prob = 0.2):
        super(GNNRecommenderwithSkipConnections, self).__init__()

        self.dropout_prob = dropout_prob

        # feature transformation layers
        self.user_feature_transform = nn.Linear(user_feature_dim, embedding_dim)
        self.product_feature_transform = nn.Linear(product_feature_dim, embedding_dim)

        # GNN layers
        self.conv1 = GCNConv(embedding_dim, embedding_dim)
        self.bn1 = nn.BatchNorm1d(embedding_dim)  # Batch normalization layer
        self.conv2 = GCNConv(embedding_dim, embedding_dim)
        self.bn2 = nn.BatchNorm1d(embedding_dim)  # Batch normalization layer

        # prediction layer
        self.predictor = nn.Linear(embedding_dim, 1)

    def forward(self, edge_index, user_features, product_features):

        # Obtains raw user and product indices
        user_indices = edge_index[0]
        product_indices = edge_index[1] - user_features.shape[0]

        # transform features
        user_x = self.user_feature_transform(user_features)
        product_x = self.product_feature_transform(product_features)

        # combine user and product features
        x = torch.cat([user_x, product_x], dim=0)

        # combined edge index for message passing
        combined_edge_index = torch.cat([
            edge_index,
            torch.stack([edge_index[1], edge_index[0]], dim=0)
        ], dim=1)

        # GNN layers
        # Layer 1: Add skip connection from the input features
        x1 = F.relu(self.bn1(self.conv1(x, combined_edge_index)))
        x1 = F.dropout(x1, p=self.dropout_prob, training=self.training)
        x1 = x1 + x  # Skip connection from input

        x2 = self.bn2(self.conv2(x1, combined_edge_index))
        x2 = x2 + x1  # Skip connection from Layer 1

        user_embeddings = x2[:len(user_x)]
        product_embeddings = x2[len(user_x):]

        # embeddings for the specific user-product pairs in edge_index
        user_emb = user_embeddings[user_indices]
        product_emb = product_embeddings[product_indices]

        # Normalize the embeddings
        user_emb_norm = F.normalize(user_emb, p=2, dim=-1)
        product_emb_norm = F.normalize(product_emb, p=2, dim=-1)
        
        # Compute element-wise dot product
        dot_product = user_emb_norm * product_emb_norm

        # predict ratings
        predictions = self.predictor(dot_product).squeeze()

        return predictions
    
class GNNSAGERecommenderwithSkipConnections(nn.Module):
    def __init__(self, num_users, num_products, user_feature_dim, product_feature_dim, embedding_dim=64, dropout_prob = 0.2):
        super(GNNSAGERecommenderwithSkipConnections, self).__init__()

        self.dropout_prob = dropout_prob

        # feature transformation layers
        self.user_feature_transform = nn.Linear(user_feature_dim, embedding_dim)
        self.product_feature_transform = nn.Linear(product_feature_dim, embedding_dim)

        # GNN layers
        self.conv1 = SAGEConv(embedding_dim, embedding_dim)
        self.bn1 = nn.BatchNorm1d(embedding_dim)  # Batch normalization layer
        self.conv2 = SAGEConv(embedding_dim, embedding_dim)
        self.bn2 = nn.BatchNorm1d(embedding_dim)  # Batch normalization layer

        # prediction layer
        self.predictor = nn.Linear(embedding_dim, 1)

    def forward(self, edge_index, user_features, product_features):

        # Obtains raw user and product indices
        user_indices = edge_index[0]
        product_indices = edge_index[1] - user_features.shape[0]

        # transform features
        user_x = self.user_feature_transform(user_features)
        product_x = self.product_feature_transform(product_features)

        # combine user and product features
        x = torch.cat([user_x, product_x], dim=0)

        # combined edge index for message passing
        combined_edge_index = torch.cat([
            edge_index,
            torch.stack([edge_index[1], edge_index[0]], dim=0)
        ], dim=1)

        # GNN layers
        # Layer 1: Add skip connection from the input features
        x1 = F.relu(self.bn1(self.conv1(x, combined_edge_index)))
        x1 = F.dropout(x1, p=self.dropout_prob, training=self.training)
        x1 = x1 + x  # Skip connection from input

        x2 = self.bn2(self.conv2(x1, combined_edge_index))
        x2 = x2 + x1  # Skip connection from Layer 1

        user_embeddings = x2[:len(user_x)]
        product_embeddings = x2[len(user_x):]

        # embeddings for the specific user-product pairs in edge_index
        user_emb = user_embeddings[user_indices]
        product_emb = product_embeddings[product_indices]

        # Normalize the embeddings
        user_emb_norm = F.normalize(user_emb, p=2, dim=-1)
        product_emb_norm = F.normalize(product_emb, p=2, dim=-1)
        
        # Compute element-wise dot product
        dot_product = user_emb_norm * product_emb_norm

        # predict ratings
        predictions = self.predictor(dot_product).squeeze()

        return predictions

class GAT_model(torch.nn.Module):
    def __init__(self, num_users, num_products, user_feature_dim, product_feature_dim, 
                 hidden_dim=64, dropout_prob=0.2):  # Added dropout_prob
        super().__init__()
        self.dropout_prob = dropout_prob

        self.user_feature_transform = nn.Linear(user_feature_dim, hidden_dim)
        self.product_feature_transform = nn.Linear(product_feature_dim, hidden_dim)

        self.conv1 = GATConv(hidden_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)

        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, edge_index, user_features, product_features):

        # Obtains raw user and product indices
        user_indices = edge_index[0]
        product_indices = edge_index[1] - user_features.shape[0]

        # transform features
        user_x = self.user_feature_transform(user_features)
        product_x = self.product_feature_transform(product_features)

        # combine user and product features
        x = torch.cat([user_x, product_x], dim=0)

        # combined edge index for message passing
        combined_edge_index = torch.cat([
            edge_index,
            torch.stack([edge_index[1], edge_index[0]], dim=0)
        ], dim=1)

        # GNN layers
        x = F.relu(self.conv1(x, combined_edge_index))
        x = F.dropout(x, p=self.dropout_prob, training=self.training)  # Modified
        x = self.conv2(x, combined_edge_index)

        user_embeddings = x[:len(user_x)]
        product_embeddings = x[len(user_x):]

        # embeddings for the specific user-product pairs in edge_index
        user_emb = user_embeddings[user_indices]
        product_emb = product_embeddings[product_indices]

        # Normalize the embeddings
        user_emb_norm = F.normalize(user_emb, p=2, dim=-1)
        product_emb_norm = F.normalize(product_emb, p=2, dim=-1)
        
        # Compute element-wise dot product
        dot_product = user_emb_norm * product_emb_norm

        # predict ratings
        predictions = self.predictor(dot_product).squeeze()

        return predictions

class GATv2_model(torch.nn.Module):
    def __init__(self, num_users, num_products, user_feature_dim, product_feature_dim, 
                 hidden_dim=64, heads=2, dropout_prob=0.2):  # Added dropout_prob
        super().__init__()
        self.dropout_prob = dropout_prob

        self.user_feature_transform = nn.Linear(user_feature_dim, hidden_dim)
        self.product_feature_transform = nn.Linear(product_feature_dim, hidden_dim)

        self.conv1 = GATv2Conv(hidden_dim, hidden_dim, heads=heads, concat=False)
        self.conv2 = GATv2Conv(hidden_dim, hidden_dim, heads=heads, concat=False)

        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, edge_index, user_features, product_features):
      # Obtains raw user and product indices
      user_indices = edge_index[0]
      product_indices = edge_index[1] - user_features.shape[0]

      # transform features
      user_x = self.user_feature_transform(user_features)
      product_x = self.product_feature_transform(product_features)

      # combine user and product features
      x = torch.cat([user_x, product_x], dim=0)

      # combined edge index for message passing
      combined_edge_index = torch.cat([
          edge_index,
          torch.stack([edge_index[1], edge_index[0]], dim=0)
      ], dim=1)

      # GNN layers
      x = F.relu(self.conv1(x, combined_edge_index))
      x = F.dropout(x, p=self.dropout_prob, training=self.training)  # Modified
      x = self.conv2(x, combined_edge_index)

      user_embeddings = x[:len(user_x)]
      product_embeddings = x[len(user_x):]

      # embeddings for the specific user-product pairs in edge_index
      user_emb = user_embeddings[user_indices]
      product_emb = product_embeddings[product_indices]

      # Normalize the embeddings
      user_emb_norm = F.normalize(user_emb, p=2, dim=-1)
      product_emb_norm = F.normalize(product_emb, p=2, dim=-1)
      
      # Compute element-wise dot product
      dot_product = user_emb_norm * product_emb_norm

      # predict ratings
      predictions = self.predictor(dot_product).squeeze()

      return predictions