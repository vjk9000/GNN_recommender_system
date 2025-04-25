import pandas as pd
import numpy as np
import torch

from utils.setup_nodes import create_edge_lists, create_user_df
from utils.graph_helpers import train_model, plot_train_val_loss, final_evaluation, make_df
from utils.graph_model import Base_GNN_Model
from utils.general import seed_everything
from utils.predictions import recommend_products, pretty_print_recomendations, get_top_k_preds

seed_everything()

user_split = "train_test_valid"
prod_embed_name = "meta_features_512"
user_embed_name = "user_reviews_features_512"

data_dir = "data"
product_dir = "full_data"
embedding_dir = "embedding"
results_folder = "baseline_with_hooks"

device = "cuda"

product_cols = ["parent_asin", "average_rating", "rating_number"]
user_cols = ["user_id", "rating_mean", "rating_count", "helpful_vote_mean", "helpful_vote_gte_1", "verified_purchase_mean", "last_active_in_days_min",
            "last_active_in_days_max", "word_count_mean"]
edge_cols = ["user_id", "parent_asin", "rating"]

product_df = pd.read_parquet(f"{data_dir}/{product_dir}/product_df.parquet", columns = product_cols)
train_user_df = pd.read_parquet(f"{data_dir}/{user_split}_split/train_agg.parquet", columns = user_cols)
train_user_edges = pd.read_parquet(f"{data_dir}/{user_split}_split/train.parquet", columns = edge_cols)

if user_split == "train_test_valid":
    test_user_df = pd.read_parquet(f"{data_dir}/{user_split}_split/valid_agg.parquet", columns = user_cols)
    test_user_edges = pd.read_parquet(f"{data_dir}/{user_split}_split/valid.parquet", columns = edge_cols)
else:
    test_user_df = pd.read_parquet(f"{data_dir}/{user_split}_split/test_agg.parquet", columns = user_cols)
    test_user_edges = pd.read_parquet(f"{data_dir}/{user_split}_split/test.parquet", columns = edge_cols)

product_embedding = torch.load(f"{data_dir}/{embedding_dir}/product/{prod_embed_name}.pt")
train_user_embedding = torch.load(f"{data_dir}/{embedding_dir}/{user_split}_split/train_{user_embed_name}.pt")
if user_split == "train_test_valid":
    test_user_embedding = torch.load(f"{data_dir}/{embedding_dir}/{user_split}_split/valid_{user_embed_name}.pt")
else:
    test_user_embedding = torch.load(f"{data_dir}/{embedding_dir}/{user_split}_split/test_{user_embed_name}.pt")

train_user_df["embedding"] = list(train_user_embedding.numpy())
test_user_df["embedding"] = list(test_user_embedding.numpy())

user_df = create_user_df(train_user_df, test_user_df)

# Set up id mapping
offset = user_df.user_id.nunique()
user_id_to_idx = {unique_id : idx for idx, unique_id in enumerate(user_df.user_id.unique())}
prod_id_to_idx = {unique_id : offset + idx for idx, unique_id in enumerate(product_df.parent_asin.unique())}

# Add to df
product_df["prod_idx"] = product_df.parent_asin.apply(lambda x: prod_id_to_idx[x])
train_user_edges["user_idx"] = train_user_edges.user_id.apply(lambda x: user_id_to_idx[x])
test_user_edges["user_idx"] = test_user_edges.user_id.apply(lambda x: user_id_to_idx[x])
train_user_edges["prod_idx"] = train_user_edges.parent_asin.apply(lambda x: prod_id_to_idx[x])
test_user_edges["prod_idx"] = test_user_edges.parent_asin.apply(lambda x: prod_id_to_idx[x])

# Concat product nodes 
product_nodes = torch.cat([torch.tensor(product_df.drop(["parent_asin", "prod_idx"], axis = 1).to_numpy()), product_embedding], dim = 1)

# concat user nodes 
user_embed = torch.tensor(np.vstack(user_df["embedding"].values))
user_info = torch.tensor(user_df.drop(["user_id", "embedding"], axis = 1).to_numpy())
user_nodes = torch.cat([user_info, user_embed], dim = 1)

# Create edge list
train_edge_index, train_edge_weights = create_edge_lists(train_user_edges)
test_edge_index, test_edge_weights = create_edge_lists(train_user_edges)

product_nodes = product_nodes.type(torch.float).to(device)
user_nodes = user_nodes.type(torch.float).to(device)
train_edge_index = train_edge_index.to(device)
train_edge_weights = train_edge_weights.to(device)
test_edge_index = test_edge_index.to(device)
test_edge_weights = test_edge_weights.to(device)

# Set up model features (fixed)
num_users = len(user_df)
num_products = len(product_df)
user_feature_dim = user_nodes.shape[1]
product_feature_dim = product_nodes.shape[1]

num_epochs = 2000

# Config ls 
config_ls = []

embedding_dim_ls = [16, 32, 64, 128, 256]
learning_rate_ls = [0.01, 1e-3, 5e-4, 1e-4]
dropout_prob_ls = [0.1, 0.2, 0.3, 0.5]

for embedding_dim in embedding_dim_ls:
    for learning_rate in learning_rate_ls:
        for dropout_prob in dropout_prob_ls:
            config_ls.append((embedding_dim, learning_rate, dropout_prob))

train_loss_ls = []
test_loss_ls = []
full_test_loss_ls = []
best_epoch_ls = []
best_test_loss_ls = []

for config in config_ls:
    embedding_dim, learning_rate, dropout_prob = config
    model = Base_GNN_Model(num_users, num_products, user_feature_dim, product_feature_dim, embedding_dim, dropout_prob)
    model.to(device=device)
    train_loss, test_loss, best_model, best_epoch = train_model(model, train_edge_index, train_edge_weights, test_edge_index, test_edge_weights,
                                                    user_nodes, product_nodes, num_epochs = num_epochs, print_progress=False, lr = learning_rate, 
                                                    give_epoch= True)
    
    full_test_loss, _ = final_evaluation(model, test_edge_index, test_edge_weights, user_nodes, product_nodes, device, print_test=False)
    model.load_state_dict(best_model)
    best_test_loss, _ = final_evaluation(model, test_edge_index, test_edge_weights, user_nodes, product_nodes, device, print_test=False)
    train_loss_ls.append(train_loss)
    test_loss_ls.append(test_loss)
    full_test_loss_ls.append(full_test_loss.item())
    best_epoch_ls.append(best_epoch)
    best_test_loss_ls.append(best_test_loss.item())

results_df = make_df(config_ls, ["embedding_dim", "learning_rate", "dropout_prob"], train_loss_ls, test_loss_ls, full_test_loss_ls, best_epoch_ls, best_test_loss_ls)

results_df.to_parquet(f"results/{results_folder}/hyper_param_tuning_2000.parquet")