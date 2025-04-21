import torch
import pandas as pd

def create_edge_lists(df):
    from_node = df["user_idx"].to_list()
    to_node = df["prod_idx"].to_list()
    edge_list =  torch.tensor([from_node, to_node], dtype=torch.long)
    edge_weight = torch.tensor(df.rating.values, dtype=torch.float)
    return edge_list, edge_weight

# def combine_edges(edge_list):
#     user_indices = edge_list[0].cpu()
#     product_indices = edge_list[1].cpu()
#     forward_edge_index = torch.stack([user_indices, product_indices], dim=0)
#     reverse_edge_index = torch.stack([product_indices, user_indices], dim=0)
#     combined_edge_index = torch.cat([forward_edge_index, reverse_edge_index], dim=1)
#     return combined_edge_index

def create_user_df(train_df, test_df):
    additional_test_users = test_df[~test_df.user_id.isin(train_df.user_id)].copy()
    embedding_shape = test_df["embedding"].iloc[0].shape[0]
    additional_test_users["embedding"] = list(torch.zeros((len(additional_test_users), embedding_shape)).numpy())
    additional_test_users["word_count_mean"] = 0
    additional_test_users.iloc[:, 1:-2] = train_df.iloc[:, 1:-2].median()
    df = pd.concat([train_df.copy(), additional_test_users.copy()])
    return df