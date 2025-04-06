
import torch

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
    product_nodes = list(range(len(product_mapping)))

    recommendation_edge_index = torch.tensor([user_nodes, product_nodes], dtype=torch.long).to(device)

    # predictions
    with torch.no_grad():
        predictions = model(recommendation_edge_index, user_features, product_features)

    # generate top-k product
    _, top_indices = torch.topk(predictions, k=top_k)

    # indices back to product asin
    reverse_mapping = {idx: asin for asin, idx in product_mapping.items()}
    recommended_products = [reverse_mapping[idx.item()] for idx in top_indices]

    return recommended_products, predictions

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