"""
This script defines function necessary for calculating NDCG scores
"""

import torch
import copy

def ndcg(predictions: torch.sparse_coo_tensor, ground_truth: torch.sparse_coo_tensor, k: int) -> torch.Tensor:

    """
    This function is too memory intensive to compute on full dataset due to sparse to dense matrix conversion.
    Tried sparse NDCG computation but takes too long due to for loop.
    Landed on batching + dense computation by looping batches over this function.
    """

    # Convert sparse tensors to dense tensors
    predictions_dense = predictions.to_dense()
    ground_truth_dense = ground_truth.to_dense()

    # Sort predictions and ground truth by predicted scores (descending order)
    _, indices = torch.topk(predictions_dense, k, dim=1, largest=True, sorted=True)
    sorted_ground_truth = torch.gather(ground_truth_dense, dim=1, index=indices)

    # Compute Discounted Cumulative Gain (DCG)
    ranks = torch.arange(1, k + 1, device=predictions.device).float()
    discount = 1.0 / torch.log2(ranks + 1)
    dcg = (sorted_ground_truth * discount).sum(dim=1)

    # Compute Ideal Discounted Cumulative Gain (IDCG)
    ideal_sorted_ground_truth, _ = torch.topk(ground_truth_dense, k, dim=1, largest=True, sorted=True)
    idcg = (ideal_sorted_ground_truth * discount).sum(dim=1)

    # Compute NDCG
    ndcg = dcg / idcg

    return ndcg


# def batch_input_variables(predictions: torch.sparse_coo_tensor, ground_truth: torch.sparse_coo_tensor, batch_size: int = 10000) -> torch.Tensor:

#     """
#     Obtains batches of data to loop over
#     """

#     # Initiate list to store batched variables
#     batched_predictions_list = []
#     batched_ground_truth_list = []
    
#     # Custom batching logic (Note: Input predictions and ground truth to be sorted by user index, which is already the case)
#     for data_no in range(2):
#         sparse_matrix = [predictions, ground_truth][data_no]
#         previous_index = 0
#         previous_selected_value = -1
#         indices = sparse_matrix.indices()
#         for i in range(batch_size, round(len(indices[0])/batch_size)*batch_size+1, batch_size):        
#             selected_index = min(len(indices[0])-1, i)
#             selected_value = indices[0][selected_index].item()
#             mask = indices[0] == selected_value
#             matching_indices = torch.nonzero(mask, as_tuple=False).squeeze(1)
#             current_index = max(matching_indices).item() + 1
#             if previous_index < current_index:
#                 sparse_matrix_batched_indices = copy.deepcopy(sparse_matrix.indices()[:, previous_index:current_index])
#                 sparse_matrix_batched_values = copy.deepcopy(sparse_matrix.values()[previous_index:current_index])
#                 sparse_matrix_batched_indices[0] = sparse_matrix_batched_indices[0] - previous_selected_value - 1
#                 sparse_matrix_batched_indices[1] = sparse_matrix_batched_indices[1] - min(sparse_matrix_batched_indices[1])
#                 sparse_matrix_batched = torch.sparse_coo_tensor(sparse_matrix_batched_indices, sparse_matrix_batched_values).coalesce()
#                 if data_no == 0:
#                     batched_predictions_list.append(sparse_matrix_batched)
#                 if data_no == 1:
#                     batched_ground_truth_list.append(sparse_matrix_batched)
#                 previous_index = current_index
#                 previous_selected_value = selected_value
    
#     return batched_predictions_list, batched_ground_truth_list

import torch

def batch_input_variables(
    predictions: torch.Tensor,
    ground_truth: torch.Tensor,
    batch_size: int = 10000,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
):
    """
    Batches sparse prediction and ground truth tensors by user index (row 0 in indices).
    Moves tensors to specified device (GPU/CPU).
    Returns lists of batched sparse tensors.

    This code was optimised from the above original implementation with the help of chat gpt

    """
    def batch_sparse_tensor(sparse_tensor):
        # Move to device
        sparse_tensor = sparse_tensor.coalesce().to(device)
        indices = sparse_tensor.indices()
        values = sparse_tensor.values()
        user_indices = indices[0]

        # Get unique users and counts
        unique_users, user_counts = torch.unique_consecutive(user_indices, return_counts=True)

        # Cumulative sum to determine split points
        user_offsets = torch.cumsum(user_counts, dim=0)
        start_idx = 0

        batched_list = []
        for batch_start in range(0, len(unique_users), batch_size):
            batch_end = min(batch_start + batch_size, len(unique_users))

            # Determine the index range in the original sparse tensor
            end_idx = user_offsets[batch_end - 1].item()
            batch_indices = indices[:, start_idx:end_idx].clone()
            batch_values = values[start_idx:end_idx].clone()

            # Normalize indices
            user_offset = batch_indices[0, 0].item()
            item_offset = batch_indices[1].min().item()
            batch_indices[0] -= user_offset
            batch_indices[1] -= item_offset

            # Create sparse batch tensor
            batch_tensor = torch.sparse_coo_tensor(
                batch_indices,
                batch_values,
                device=device
            ).coalesce()

            batched_list.append(batch_tensor)
            start_idx = end_idx

        return batched_list

    batched_predictions = batch_sparse_tensor(predictions)
    batched_ground_truth = batch_sparse_tensor(ground_truth)

    return batched_predictions, batched_ground_truth



def batch_ndcg(predictions: torch.sparse_coo_tensor, ground_truth: torch.sparse_coo_tensor, k: int, batch_size: int = 10000, verbose=False) -> torch.Tensor:

    """
    NDCG calculations that incorporates batching
    """

    # Predictions and ground truth should have the same size for batching to work
    assert predictions.shape == ground_truth.shape
    
    # First, batch up the predictions and ground truth so that we can do dense matrix computations
    batched_predictions_list, batched_ground_truth_list = batch_input_variables(predictions, ground_truth, batch_size)

    # Now, use NDCG to calculate NDCG for each user and combine them to calculate overall NDCG
    ndcg_all = torch.empty(0)
    for i in range(len(batched_predictions_list)):
        #if verbose:
        print(f"Running batch number {i+1} out of {len(batched_predictions_list)}...")
        batch_ndcg_scores = ndcg(batched_predictions_list[i], batched_ground_truth_list[i], k)
        ndcg_all = torch.cat([ndcg_all.cpu(), batch_ndcg_scores.cpu()], dim=0)
        

    return ndcg_all


def calculate_ndcg_scores(model, test_edge_index, test_edge_weights, user_features, product_features, k=10, batch_size=10000):
    """
    Calculates NDCG scores.
    Note: This is currently a bit iffy, since we are only calculating NCDG for the products included in the test set (i.e., we are only recommending products in the test set, not the whole universe).
    Implementing recommendations for the whole user and product universe is possible but will be more tricky - this has not yet been implemented.
    """
    model.eval()
    with torch.no_grad():
        test_predictions = model(test_edge_index, user_features, product_features)
        test_predictions_sparse = torch.sparse_coo_tensor(test_edge_index, test_predictions).coalesce()
        test_ground_truth = torch.sparse_coo_tensor(test_edge_index, test_edge_weights).coalesce()
        ndcg_scores = batch_ndcg(test_predictions_sparse, test_ground_truth, k, batch_size)
        
    return ndcg_scores

# helper functions to recommend products with NDCG
def recommend_products_with_ndcg(model, test_edge_index, test_edge_weights, user_id, user_mapping, product_mapping, user_features, product_features, top_k=10, device = None, 
                                 batch_size=256):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model.eval()

    # Filter for relevant edges and weights
    mask = test_edge_index[0] == user_id
    matching_indices = torch.nonzero(mask, as_tuple=False).squeeze(1)
    sample_test_edge_index = test_edge_index[:, mask]
    sample_test_edge_weights = test_edge_weights[mask]

    # Create complete test edge index
    user_nodes = [user_id] * len(product_mapping)
    product_nodes = list(product_mapping.values())
    complete_test_edge_index = torch.tensor([user_nodes, product_nodes], dtype=torch.long).to(device)
    # complete_test_edge_index = torch.stack([torch.tensor([selected_user_idx]).repeat(unique_test_products), product_features], dim=0)
    
    # Create a tensor of zeros for complete_test_edge_weights
    num_edges = complete_test_edge_index.size(1)  # Number of edges (columns in complete_test_edge_index)
    complete_test_edge_weights = torch.zeros(num_edges).to(device)
    
    # Find the indices in complete_test_edge_index that match sample_test_edge_index
    complete_test_edge_index_t = complete_test_edge_index.t().to(device)  # Shape: [num_edges, 2]
    sample_test_edge_index_t = sample_test_edge_index.t().to(device)      # Shape: [num_sample_edges, 2]
    
    # Compare each edge in complete_test_edge_index with sample_test_edge_index
    complete_mask = (complete_test_edge_index_t.unsqueeze(1) == sample_test_edge_index_t).all(dim=2).any(dim=1).to(device)
    
    # Assign sample_test_edge_weights to the matching indices
    complete_test_edge_weights[complete_mask] = sample_test_edge_weights

    # predictions
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
    ndcg = calculate_ndcg_scores(model, complete_test_edge_index.to(device), complete_test_edge_weights.to(device), user_features, product_features, k=10, batch_size=batch_size)
    ndcg_mean = torch.nanmean(ndcg)

    return recommended_products, ndcg_mean, scores