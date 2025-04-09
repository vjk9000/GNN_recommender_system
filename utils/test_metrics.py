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


def batch_input_variables(predictions: torch.sparse_coo_tensor, ground_truth: torch.sparse_coo_tensor, batch_size: int = 10000) -> torch.Tensor:

    """
    Obtains batches of data to loop over
    """

    # Initiate list to store batched variables
    batched_predictions_list = []
    batched_ground_truth_list = []
    
    # Custom batching logic (Note: Input predictions and ground truth to be sorted by user index, which is already the case)
    for data_no in range(2):
        sparse_matrix = [predictions, ground_truth][data_no]
        previous_index = 0
        previous_selected_value = -1
        indices = sparse_matrix.indices()
        for i in range(batch_size, round(len(indices[0])/batch_size)*batch_size+1, batch_size):        
            selected_index = min(len(indices[0])-1, i)
            selected_value = indices[0][selected_index].item()
            mask = indices[0] == selected_value
            matching_indices = torch.nonzero(mask, as_tuple=False).squeeze(1)
            current_index = max(matching_indices).item() + 1
            if previous_index < current_index:
                sparse_matrix_batched_indices = copy.deepcopy(sparse_matrix.indices()[:, previous_index:current_index])
                sparse_matrix_batched_values = copy.deepcopy(sparse_matrix.values()[previous_index:current_index])
                sparse_matrix_batched_indices[0] = sparse_matrix_batched_indices[0] - previous_selected_value - 1
                sparse_matrix_batched_indices[1] = sparse_matrix_batched_indices[1] - min(sparse_matrix_batched_indices[1])
                sparse_matrix_batched = torch.sparse_coo_tensor(sparse_matrix_batched_indices, sparse_matrix_batched_values).coalesce()
                if data_no == 0:
                    batched_predictions_list.append(sparse_matrix_batched)
                if data_no == 1:
                    batched_ground_truth_list.append(sparse_matrix_batched)
                previous_index = current_index
                previous_selected_value = selected_value
    
    return batched_predictions_list, batched_ground_truth_list


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
        if verbose:
            print(f"Running batch number {i+1} out of {len(batched_predictions_list)}...")
        batch_ndcg_scores = ndcg(batched_predictions_list[i], batched_ground_truth_list[i], k)
        ndcg_all = torch.cat([ndcg_all, batch_ndcg_scores], dim=0)
        

    return ndcg_all


def calculate_ndcg_scores(model, best_model, test_edge_index, test_edge_weights, user_features, product_features, k=10, batch_size=10000):
    """
    Calculates NDCG scores.
    Note: This is currently a bit iffy, since we are only calculating NCDG for the products included in the test set (i.e., we are only recommending products in the test set, not the whole universe).
    Implementing recommendations for the whole user and product universe is possible but will be more tricky - this has not yet been implemented.
    """
    
    model.load_state_dict(best_model)
    model.eval()
    with torch.no_grad():
        test_predictions = model(test_edge_index, user_features, product_features)
        test_predictions_sparse = torch.sparse_coo_tensor(test_edge_index, test_predictions).coalesce()
        test_ground_truth = torch.sparse_coo_tensor(test_edge_index, test_edge_weights).coalesce()
        ndcg_scores = batch_ndcg(test_predictions_sparse, test_ground_truth, k, batch_size)
        
    return ndcg_scores