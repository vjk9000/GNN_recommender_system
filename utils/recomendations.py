    
import torch
from utils.test_metrics import calculate_ndcg_scores

# helper functions to recommend products
def recommend_products(model, user_id, user_mapping, product_mapping, user_features, product_features, top_k=10, device = None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # move to gpu
    # model.to(device)
    # user_features = user_features.to(device) 
    # product_features = product_features.to(device)

    model.eval()

    # find user id
    user_idx = user_mapping[user_id]

    # create edges between this user and all products
    user_nodes = [user_idx] * len(product_mapping)
    product_nodes = list(range(len(user_mapping), len(user_mapping) + len(product_mapping)))

    recommendation_edge_index = torch.tensor([user_nodes, product_nodes], dtype=torch.long).to(device)

    # predictions
    with torch.no_grad():
        predictions = model(recommendation_edge_index, user_features, product_features)

    # generate top-k product
    _, top_indices = torch.topk(predictions, k=top_k)
    scores = predictions[top_indices]
    top_indices += len(user_mapping)

    # indices back to product asin
    reverse_mapping = {idx: asin for asin, idx in product_mapping.items()}
    recommended_products = [reverse_mapping[idx.item()] for idx in top_indices]

    return recommended_products, scores

# helper functions to recommend products with NDCG
def recommend_products_with_ndcg(model, best_model, test_edge_index, test_edge_weights, user_id, user_mapping, product_mapping, user_features, product_features, top_k=10, device = None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model.eval()

    # Filter for relevant edges and weights
    mask = test_edge_index[0] == user_id
    matching_indices = torch.nonzero(mask, as_tuple=False).squeeze(1)
    sample_test_edge_index = test_edge_index[:, mask]
    sample_test_edge_weights = test_edge_weights[mask]

    # Create complete test edge index
    complete_test_edge_index = torch.stack([torch.tensor([selected_user_idx]).repeat(len(unique_test_products)), unique_test_products], dim=0)
    
    # Create a tensor of zeros for complete_test_edge_weights
    num_edges = complete_test_edge_index.size(1)  # Number of edges (columns in complete_test_edge_index)
    complete_test_edge_weights = torch.zeros(num_edges)
    
    # Find the indices in complete_test_edge_index that match sample_test_edge_index
    complete_test_edge_index_t = complete_test_edge_index.t()  # Shape: [num_edges, 2]
    sample_test_edge_index_t = sample_test_edge_index.t()      # Shape: [num_sample_edges, 2]
    
    # Compare each edge in complete_test_edge_index with sample_test_edge_index
    complete_mask = (complete_test_edge_index_t.unsqueeze(1) == sample_test_edge_index_t).all(dim=2).any(dim=1)
    
    # Assign sample_test_edge_weights to the matching indices
    complete_test_edge_weights[complete_mask] = sample_test_edge_weights

    # predictions
    model.load_state_dict(best_model)
    with torch.no_grad():
        predictions = model(complete_test_edge_index, user_features, product_features)

    # generate top-k product
    _, top_indices = torch.topk(predictions, k=top_k)
    scores = predictions[top_indices]
    top_indices += len(user_mapping)

    # indices back to product asin
    reverse_mapping = {idx: asin for asin, idx in product_mapping.items()}
    recommended_products = [reverse_mapping[idx.item()] for idx in top_indices]

    # Calculate NDCG score
    ndcg = calculate_ndcg_scores(model, best_model, complete_test_edge_index, complete_test_edge_weights, user_features, product_features, k=10, batch_size=5)
    ndcg_mean = torch.nanmean(ndcg)

    return recommended_products, ndcg_mean, scores

def pretty_print_recomendations(recommended_asins, product_df, col):
    for index, asin in enumerate(recommended_asins, start=1):
        print("Product {}:".format(index), product_df[product_df['parent_asin'] == asin][col].values[0])

def dcg_at_k(relevance_scores, k=10):
    """
    Compute Discounted Cumulative Gain (DCG) at rank k.
    GENERATED WITH CHAT GPT 
    
    :param relevance_scores: List or tensor of relevance scores in predicted order
    :param k: Cutoff rank (default=10)
    :return: DCG@k score
    """
    relevance_scores = torch.tensor(relevance_scores, dtype=torch.float32)
    discounts = torch.log2(torch.arange(2, len(relevance_scores) + 2, dtype=torch.float32))  # log2(i+1) for i=1,...,k
    return torch.sum(relevance_scores / discounts).item()

def paper_evaluation(model, user_idx, recomended_idx, prod_map_len, user_features, product_features, top_k = 10,device = None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model.eval()
    # model.to(device)
    # user_features = user_features.to(device)
    # product_features =  product_features.to(device)

    product_nodes = list(range(prod_map_len))

    with torch.no_grad():
        user_nodes = [user_idx for i in range(prod_map_len)] 
        recommendation_edge_index = torch.tensor([user_nodes, product_nodes], dtype=torch.long).to(device)
        predictions = model(recommendation_edge_index, user_features, product_features)
     
    _, top_indices = torch.topk(predictions, k=top_k)
    
    # recall @ 10 
    recall_10 = len(set(top_indices).intersection(set(recomended_idx))) / len(recomended_idx)

    # ndcg @ 10
    relevance_scores = [1 if x in set(recomended_idx) else 0 for x in top_indices]
    ideal_relevance_scores = [1 for x in recomended_idx]
    dcg_10 = dcg_at_k(relevance_scores, top_k)
    idcg_10 = dcg_at_k(ideal_relevance_scores, top_k)
    ndcg_10 = dcg_10 / idcg_10

    return recall_10, ndcg_10