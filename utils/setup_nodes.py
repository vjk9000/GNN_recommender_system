import torch

def create_edge_lists(df):
    from_node = df["user_idx"].to_list()
    to_node = df["prod_idx"].to_list()
    edge_list =  torch.tensor([from_node, to_node], dtype=torch.long)
    edge_weight = torch.tensor(df.rating.values, dtype=torch.float)
    return edge_list, edge_weight

def combine_edges(edge_list):
    user_indices = edge_list[0].cpu()
    product_indices = edge_list[1].cpu()
    forward_edge_index = torch.stack([user_indices, product_indices], dim=0)
    reverse_edge_index = torch.stack([product_indices, user_indices], dim=0)
    combined_edge_index = torch.cat([forward_edge_index, reverse_edge_index], dim=1)
    return combined_edge_index