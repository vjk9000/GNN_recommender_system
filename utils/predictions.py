import torch

# from utils.graph_helpers import combine_edges

# helper functions to recommend products
def recommend_products(model, user_id, user_mapping, product_mapping, user_features, product_features, reverse_mapping, top_k=10, device = None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model.eval()

    # find user id
    user_idx = user_mapping[user_id]

    # create edges between this user and all products
    user_nodes = [user_idx] * len(product_mapping)
    product_nodes = list(product_mapping.values())

    # Normal edge
    recommendation_edge_index = torch.tensor([user_nodes, product_nodes], dtype=torch.long).to(device)

    # Flip edge
    # reco_combined_egde_index = combine_edges(recommendation_edge_index)
    # reco_combined_egde_index = reco_combined_egde_index.to(device)

    # predictions
    with torch.no_grad():
        predictions = model(recommendation_edge_index, user_features, product_features)

    # generate top-k product
    _, top_indices = torch.topk(predictions, k=top_k)

    # indices back to product asin
    recommended_products = [reverse_mapping[idx.item()] for idx in top_indices]

    return recommended_products, predictions

def pretty_print_recomendations(recommended_asins, product_df, col):
    for index, asin in enumerate(recommended_asins, start=1):
        print("Product {}:".format(index), product_df[product_df['parent_asin'] == asin][col].values[0])

def get_top_k_preds(model, test_user_idx_ls, k, batch_size, user_nodes, prod_nodes, product_idx, device = None):
    model.eval()
    sol = []
    num_products = len(product_idx)
    
    if device == None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    for index in range(0, len(test_user_idx_ls), batch_size):
        test_user_ids = torch.tensor(test_user_idx_ls[index:index+batch_size], dtype=torch.long, device=device)
        test_size = test_user_ids.shape[0]
        
        # user_ids_expanded = test_user_ids.unsqueeze(1).expand(-1, num_products)
        # product_ids_expanded = product_idx.unsqueeze(0).expand(test_size, -1)
        # user_ids_flat = user_ids_expanded.reshape(-1)
        # product_ids_flat = product_ids_expanded.reshape(-1)
        # batch_test_edges = torch.stack([user_ids_flat.cpu(), product_ids_flat]).to(device)

        # Optimised with the help of Chat GPT
        user_ids_flat = test_user_ids.repeat_interleave(num_products)
        product_ids_flat = product_idx.repeat(test_size)
        batch_test_edges = torch.stack([user_ids_flat, product_ids_flat], dim=0)

        with torch.no_grad():
            scores_flat = model(batch_test_edges, user_nodes, prod_nodes)
        
        scores = scores_flat.view(test_user_ids.shape[0], num_products)
        topk_scores, topk_indices = torch.topk(scores, k, dim=1)
        sol.append(topk_indices)
    
    return sol

# def evaluate_recall(model, test_edges, k, batch_size, user_nodes, prod_nodes, product_idx, device = None):
#     model.eval()
#     sol = []
#     num_products = len(product_idx)
#     if device == None:
#         device = "cuda" if torch.cuda.is_available() else "cpu"

#     for index in range(0, test_edges.shape[1], batch_size):
#         test_user_ids = test_edges[0][index:index+batch_size]
#         test_size = test_user_ids.shape[0]
#         user_ids_expanded = test_user_ids.unsqueeze(1).expand(-1, num_products)
#         product_ids_expanded = product_idx.unsqueeze(0).expand(test_size, -1)
#         user_ids_flat = user_ids_expanded.reshape(-1)
#         product_ids_flat = product_ids_expanded.reshape(-1)
#         batch_test_edges = torch.stack([user_ids_flat.cpu(), product_ids_flat]).to(device)
#         # batch_combined_egde_index = combine_edges(batch_test_edges)
#         # batch_combined_egde_index = batch_combined_egde_index.to(device)

#         with torch.no_grad():
#             scores_flat = model(batch_test_edges, user_nodes, prod_nodes)
        
#         scores = scores_flat.view(test_user_ids.shape[0], num_products)
#         topk_scores, topk_indices = torch.topk(scores, k, dim=1)
#         true_item_ids = test_edges[1][index:index+batch_size]
#         recall_hits = (topk_indices == true_item_ids.unsqueeze(1)).any(dim=1).float().cpu().tolist()
#         sol.extend(recall_hits)
    
#     return torch.tensor(sol).mean()