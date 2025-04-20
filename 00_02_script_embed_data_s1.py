import os
import pandas as pd
import torch 

from utils.make_embedding import custom_BLaIR_text_embedding_model, save_embedding

save_dir = "data/embedding"
prod_dir = "product"
two_split = "train_test_split"
three_split = "train_test_valid_split"

# product df 
product_df = pd.read_parquet("data/full_data/product_df.parquet")

# Aggregated users 
two_split_review_train_agg = pd.read_parquet(f"data/train_test_split/train_agg.parquet")
two_split_review_test_agg = pd.read_parquet(f"data/train_test_split/test_agg.parquet")
three_split_review_train_agg = pd.read_parquet(f"data/train_test_valid_split/train_agg.parquet")
three_split_review_test_agg = pd.read_parquet(f"data/train_test_valid_split/test_agg.parquet")
three_split_review_valid_agg = pd.read_parquet(f"data/train_test_valid_split/valid_agg.parquet")

# Define

## Cols
product_col = "meta"
user_col = "reviews"

## Save dir
product_df_save_name = "meta_features_512"
two_split_review_train_agg_save_name = "train_user_reviews_features_512"
two_split_review_test_agg_save_name = "test_user_reviews_features_512"
three_split_review_train_agg_save_name = "train_user_reviews_features_512"
three_split_review_test_agg_save_name = "test_user_reviews_features_512"
three_split_review_valid_agg_save_name = "valid_user_reviews_features_512"

## model params

### blair 
model_dir = "custom_blair/massive"
batch_size = 512
max_length = 512
device = "cuda"

## make dir
os.makedirs(f"{save_dir}", exist_ok=True)
os.makedirs(f"{save_dir}/{prod_dir}", exist_ok=True)
os.makedirs(f"{save_dir}/{two_split}", exist_ok=True)
os.makedirs(f"{save_dir}/{three_split}", exist_ok=True)

# embed 
product_df_embed = custom_BLaIR_text_embedding_model(product_df[product_col].to_list(), model_dir, batch_size, max_length, device)
save_embedding(product_df_embed, f"{save_dir}/{prod_dir}/{product_df_save_name}.pt")
print(f"done with product_df_save_name")

two_split_review_train_agg_embed = custom_BLaIR_text_embedding_model(two_split_review_train_agg[user_col].to_list(), model_dir, batch_size, max_length, device)
save_embedding(two_split_review_train_agg_embed, f"{save_dir}/{two_split}/{two_split_review_train_agg_save_name}.pt")
print(f"done with two_split_review_train_agg_save_name")

two_split_review_test_agg_embed = custom_BLaIR_text_embedding_model(two_split_review_test_agg[user_col].to_list(), model_dir, batch_size, max_length, device)
save_embedding(two_split_review_test_agg_embed, f"{save_dir}/{two_split}/{two_split_review_test_agg_save_name}.pt")
print(f"done with two_split_review_test_agg_save_name")

three_split_review_train_agg_embed = custom_BLaIR_text_embedding_model(three_split_review_train_agg[user_col].to_list(), model_dir, batch_size, max_length, device)
save_embedding(three_split_review_train_agg_embed, f"{save_dir}/{three_split}/{three_split_review_train_agg_save_name}.pt")
print(f"done with three_split_review_train_agg_save_name")

three_split_review_test_agg_embed = custom_BLaIR_text_embedding_model(three_split_review_test_agg[user_col].to_list(), model_dir, batch_size, max_length, device)
save_embedding(three_split_review_test_agg_embed, f"{save_dir}/{three_split}/{three_split_review_test_agg_save_name}.pt")
print(f"done with three_split_review_test_agg_save_name")

three_split_review_valid_agg_embed = custom_BLaIR_text_embedding_model(three_split_review_valid_agg[user_col].to_list(), model_dir, batch_size, max_length, device)
save_embedding(three_split_review_valid_agg_embed, f"{save_dir}/{three_split}/{three_split_review_valid_agg_save_name}.pt")
print(f"done with three_split_review_valid_agg_save_name")