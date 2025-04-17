import os
import time

import pandas as pd
from itertools import product

from embeddings.constants import CLEANED_DATA_PATH
from embeddings.embedding_gs_wrapper import EmbeddingAndGNNWrapper


def custom_grid_search(model_class, param_grid, data_tuple, verbose=1):
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())

    param_combinations = list(product(*param_values))
    total_combinations = len(param_combinations)

    best_score = float('-inf')
    best_model = None
    best_params = None
    results = []

    if verbose > 0:
        print(f"Starting grid search with {total_combinations} combinations")

    for i, values in enumerate(param_combinations):
        params = dict(zip(param_names, values))

        if verbose > 0:
            print(f"\nCombination {i + 1}/{total_combinations}")
            print(f"Parameters: {params}")

        start_time = time.time()

        model = model_class(**params)

        if verbose > 1:
            print("Fitting model...")

        model.fit(data_tuple)

        score = model.score(data_tuple)

        duration = time.time() - start_time

        if verbose > 0:
            print(f"Score: {score:.4f} (took {duration:.2f}s)")

        results.append({
            'params': params.copy(),
            'score': score,
            'time': duration
        })

        # Track best model
        if score > best_score:
            best_score = score
            best_params = params.copy()
            best_model = model

    if verbose > 0:
        print("\nGrid search completed")
        print(f"Best parameters: {best_params}")
        print(f"Best score: {best_score:.4f}")

    return {
        'best_model': best_model,
        'best_params': best_params,
        'best_score': best_score,
        'all_results': results
    }


def gs_embeddings():
    user_features_numeric_agg = pd.read_parquet(f"{CLEANED_DATA_PATH}/user_features_numeric_agg.parquet")
    user_features_string_agg = pd.read_parquet(f"{CLEANED_DATA_PATH}/user_features_string_agg.parquet")
    product_features_numeric = pd.read_parquet(f"{CLEANED_DATA_PATH}/product_features_numeric.parquet")
    product_features_string = pd.read_parquet(f"{CLEANED_DATA_PATH}/product_features_string.parquet")

    user_features_string_agg = user_features_string_agg
    user_features_numeric_agg = user_features_numeric_agg
    product_features_string = product_features_string
    product_features_numeric = product_features_numeric

    X_custom = (
        user_features_numeric_agg,
        user_features_string_agg,
        product_features_numeric,
        product_features_string,
    )

    # will be passed into init of wrapper
    param_grid = {
        "pooling": ["cls"],
        "max_length": [2],
        "embedding_model_name": ["E5",]
    }

    results = custom_grid_search(
        model_class=EmbeddingAndGNNWrapper,
        param_grid=param_grid,
        data_tuple=X_custom,
        verbose=3
    )

    print(results)
    pd.DataFrame(results).to_csv(f'results.csv')


if __name__ == "__main__":
    gs_embeddings()
