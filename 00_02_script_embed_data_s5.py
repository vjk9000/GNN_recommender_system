import os
import pandas as pd
import torch 

from utils.make_embedding import BLaIR_roberta_base_text_embedding_model, save_embedding, custom_BLaIR_text_embedding_model

save_dir = "data/embedding"
prod_dir = "product"

# product df 
product_df = pd.read_parquet("data/full_data/product_df.parquet")

### blair 
model_dir = "custom_blair/massive"
batch_size = 512
max_length = 512
device = "cuda"

## make dir
os.makedirs(f"{save_dir}", exist_ok=True)
os.makedirs(f"{save_dir}/{prod_dir}", exist_ok=True)

# embed 
product_df_embed = BLaIR_roberta_base_text_embedding_model(product_df["meta_txf"].to_list(), batch_size, max_length, device)
save_embedding(product_df_embed, f"{save_dir}/{prod_dir}/meta_cleaned_features_base.pt")
del product_df_embed
print(f"done with meta_features_base")

product_df_embed = custom_BLaIR_text_embedding_model(product_df["meta_txf"].to_list(), model_dir, batch_size, max_length, device)
save_embedding(product_df_embed, f"{save_dir}/{prod_dir}/meta_cleaned_features_512.pt")
del product_df_embed
print(f"done with meta_features_512")