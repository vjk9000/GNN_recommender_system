from utils.graph_helpers import make_df, single_grid_search
from utils.general import seed_everything

import os

seed_everything()

user_split = "train_test_valid"

data_dir = "data"
product_dir = "full_data"
embedding_dir = "embedding"
results_folder = "embedding_grid_search"

device = "cuda"

product_cols = ["parent_asin", "average_rating", "rating_number"]
user_cols = ["user_id", "rating_mean", "rating_count", "helpful_vote_mean", "helpful_vote_gte_1", "verified_purchase_mean", "last_active_in_days_min",
            "last_active_in_days_max", "word_count_mean"]
edge_cols = ["user_id", "parent_asin", "rating"]

train_loss_ls = []
test_loss_ls = []
full_test_loss_ls = []
best_epoch_ls = []
best_test_loss_ls = []
config_ls = []

model_choice_ls = ["base", "512"]
prod_embed_ls = ["title", "description", "features", "details", 
                 "meta", "meta_cleaned",
                 "combined_title_details", "combined_title_description", "combined_title_features", 
                 "combined_all"]

for model_choice in model_choice_ls:
    for prod_embed in prod_embed_ls:
        train_loss, test_loss, final_test_loss, best_epoch, best_test_loss = single_grid_search(model_choice, prod_embed, data_dir, product_dir, 
                                                                                                embedding_dir, user_split, product_cols, user_cols, 
                                                                                                edge_cols, device)
        config_ls.append((model_choice, prod_embed))
        train_loss_ls.append(train_loss)
        test_loss_ls.append(test_loss)
        full_test_loss_ls.append(final_test_loss.item())
        best_epoch_ls.append(best_epoch)
        best_test_loss_ls.append(best_test_loss.item())

results_df = make_df(config_ls, ["embed_model", "prod_col"], train_loss_ls, test_loss_ls, full_test_loss_ls, best_epoch_ls, best_test_loss_ls)

os.makedirs(f"results/{results_folder}", exist_ok=True)
results_df.to_parquet(f"results/{results_folder}/grid_search_results_seeded.parquet")

print(results_df.sort_values("best_test_loss"))