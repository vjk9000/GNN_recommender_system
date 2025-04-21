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
product_df_embed = BLaIR_roberta_base_text_embedding_model(product_df["title_txf"].to_list(), batch_size, max_length, device)
save_embedding(product_df_embed, f"{save_dir}/{prod_dir}/title_features_base.pt")
del product_df_embed
print(f"done with title feature base")

product_df_embed = BLaIR_roberta_base_text_embedding_model(product_df["title_txf"].to_list(), batch_size, max_length, device)
save_embedding(product_df_embed, f"{save_dir}/{prod_dir}/details_features_base.pt")
del product_df_embed
print(f"done with details features base")

product_df_embed = custom_BLaIR_text_embedding_model(product_df["title_txf"].to_list(), model_dir, batch_size, max_length, device)
save_embedding(product_df_embed, f"{save_dir}/{prod_dir}/title_features_512.pt")
del product_df_embed
print(f"done with title_features_512")

product_df_embed = custom_BLaIR_text_embedding_model(product_df["title_txf"].to_list(), model_dir, batch_size, max_length, device)
save_embedding(product_df_embed, f"{save_dir}/{prod_dir}/details_features_512.pt")
del product_df_embed
print(f"done with details_features_512")

# Combine the embeddings 
for model_type in ["base", "512"]:
    title = torch.load(f"{save_dir}/{prod_dir}/title_features_{model_type}.pt")
    details = torch.load(f"{save_dir}/{prod_dir}/details_features_{model_type}.pt")
    combined = torch.cat([title, details], dim = 1)
    save_embedding(combined, f"{save_dir}/{prod_dir}/combined_title_details_features_{model_type}.pt")
    del title, details, combined
    print(f"combine done for {model_type}")
