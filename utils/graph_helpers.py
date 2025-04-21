import copy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import seaborn as sns

from utils.setup_nodes import create_edge_lists, create_user_df
from utils.graph_model import Base_GNN_Model
from utils.general import seed_everything

def plot_train_val_loss(train_loss, validation_loss = None, figsize = (6.5, 3.4), xlabel = "Epoch", ylabel = "Loss", 
                        title = 'Training and Validation Loss', grid = True):
    plt.figure(figsize=figsize)
    plt.plot(range(len(train_loss)), train_loss, label='Training Loss')
    if validation_loss is not None:
        plt.plot(range(len(validation_loss)), validation_loss, label='Validation Loss')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(grid)
    plt.show()
    return None

def plot_actual_vs_predicted_ratings(test_predictions, test_edge_weights, figsize = (6.5, 3.4)):
    plt.figure(figsize=figsize)

    # Plot
    sns.kdeplot(test_predictions, color='blue', fill=True, label="Predicted")
    sns.kdeplot(test_edge_weights, color="orange", fill=True, label="Actual")
    
    # Add labels and title
    plt.title("Density Plot of Predicted and Actual Ratings")
    plt.xlabel("Rating")
    plt.ylabel("Density")
    plt.legend()
    
    # Show the plot
    plt.show()

def train_model(model, train_edge_index, train_edge_weights, test_edge_index, test_edge_weights, user_features, product_features, 
                num_epochs=100, lr=0.01, optimiser = None, device = None, print_progress = False, print_freq = 10, give_epoch = False):
    
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if optimiser is None:
        optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    
    # lists to store loss values for plotting
    train_losses = []
    test_losses = []

    # Hold for the best config 
    best_model_state = model.state_dict()
    best_valid_loss = float("inf")
    best_model_epoch = None

    # # Make the combined edges 
    # train_combined_egde_index = combine_edges(train_edge_index)
    # test_combined_egde_index = combine_edges(test_edge_index)
    # train_combined_egde_index = train_combined_egde_index.to(device)
    # test_combined_egde_index = test_combined_egde_index.to(device)

    # training loop across epochs
    for epoch in range(1, num_epochs + 1):
        # training
        model.train()
        optimiser.zero_grad()

        # forward pass
        train_predictions = model(train_edge_index, user_features, product_features)

        # MSE 
        train_loss = nn.functional.mse_loss(train_predictions, train_edge_weights)

        # backward pass and optimisation
        train_loss.backward()
        optimiser.step()
        
        # training complete move to validation 
        model.eval()
        with torch.no_grad():
            valid_predictions = model(test_edge_index, user_features, product_features)
            valid_loss = nn.functional.mse_loss(valid_predictions, test_edge_weights)
    
        # Append losses 
        train_losses.append(train_loss.item())
        test_losses.append(valid_loss.item())

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_model_state = copy.deepcopy(model.state_dict())
            best_model_epoch = epoch

        # print progress
        if epoch % print_freq == 0 and print_progress:
            print(f'Epoch: {epoch}, Train Loss: {train_loss:.4f}, Val Loss: {valid_loss:.4f}, best model epoch: {best_model_epoch}')
    
    if give_epoch:
        return train_losses, test_losses, best_model_state, best_model_epoch
    return train_losses, test_losses, best_model_state

def train_model_without_test(model, train_edge_index, train_edge_weights, user_features, product_features,
                             num_epochs=100, lr=0.01, optimiser = None, device = None, print_progress = False, print_freq = 10):
    
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if optimiser is None:
        optimiser = torch.optim.Adam(model.parameters(), lr=lr)

    # Safety check
    train_losses = []

    # # Make the combined edges 
    # train_combined_egde_index = combine_edges(train_edge_index)
    # train_combined_egde_index = train_combined_egde_index.to(device)

    # training loop across epochs
    for epoch in range(1, num_epochs + 1):
        # training
        model.train()
        optimiser.zero_grad()

        # forward pass
        train_predictions = model(train_edge_index, user_features, product_features)

        # MSE 
        train_loss = nn.functional.mse_loss(train_predictions, train_edge_weights)

        # backward pass and optimisation
        train_loss.backward()
        optimiser.step()

        # print progress
        if epoch % print_freq == 0 and print_progress:
            print(f'Epoch: {epoch}, Train Loss: {train_loss:.4f}')

        train_losses.append(train_loss.item())
    
    return train_losses

def final_evaluation(model, test_edge_index, test_edge_weights, user_features, product_features, device = None, plot = False, print_test = True):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # # Make the combined edges 
    # test_combined_egde_index = combine_edges(test_edge_index)
    # test_combined_egde_index = test_combined_egde_index.to(device)

    # Final evaluation on test set
    model.eval()
    with torch.no_grad():
        test_predictions = model(test_edge_index, user_features, product_features)
        test_loss = nn.functional.mse_loss(test_predictions, test_edge_weights)

    if plot:
        plot_actual_vs_predicted_ratings(test_predictions.cpu(), test_edge_weights.cpu())
    
    if print_test:
        print(f"Test loss: {test_loss:.4f}")
    else:
        return test_loss

def plot_weights_heatmap_and_density(weights, attribute_key):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Heatmap (subplot 1)
    sns.heatmap(weights, annot=False, cmap="viridis", ax=axes[0])
    axes[0].set_title(f"Heatmap of {attribute_key}")
    axes[0].set_xlabel("Input Features")
    axes[0].set_ylabel("Output Features")

    # Density plot (subplot 2)
    sns.kdeplot(weights.flatten(), fill=True, color="blue", ax=axes[1])
    axes[1].set_title(f"Density Plot of {attribute_key}")
    axes[1].set_xlabel("Weight Value")
    axes[1].set_ylabel("Density")

    # Adjust layout and show the plot
    plt.tight_layout()
    plt.show()

def plot_activation_heatmap_and_density(activations_of_interest):
    # Create a figure with 2 subplots: one for the heatmap and one for the density plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Heatmap (subplot 1)
    sns.heatmap(activations_of_interest, vmin=-100, vmax=100, annot=False, cmap="viridis", ax=axes[0])
    axes[0].set_title("Activations")
    axes[0].set_xlabel("Features")
    axes[0].set_ylabel("Users")
    
    # Density plot (subplot 2)
    sns.kdeplot(activations_of_interest.flatten(), fill=True, color="blue", ax=axes[1])
    axes[1].set_title("Density Plot")
    axes[1].set_xlabel("Activations")
    axes[1].set_ylabel("Density")
    axes[1].set_xlim(-10000, 10000)
    
    # Adjust layout and show the plot
    plt.tight_layout()
    plt.show()

def make_df(config_ls, config_ls_names, train_loss_ls, test_loss_ls, full_test_loss_ls, best_epoch_ls, best_test_loss_ls):
    df = pd.DataFrame()
    for index, name in enumerate(config_ls_names):
        df[name] = [x[index] for x in config_ls]
    df["train_loss"] = train_loss_ls
    df["test_loss"] = test_loss_ls
    df["final_test_loss"] = full_test_loss_ls
    df["best_epoch"] = best_epoch_ls
    df["best_test_loss"] = best_test_loss_ls
    return df

def single_grid_search(embed_tye, prod_embed, data_dir, product_dir, embedding_dir, user_split, product_cols, user_cols, edge_cols, device, seed = 42):
    """
    A bit of inefficiency with the multiple reads but done just for simplicity
    """

    # set seed
    seed_everything(seed)

    # Load data
    product_df = pd.read_parquet(f"{data_dir}/{product_dir}/product_df.parquet", columns = product_cols)
    train_user_df = pd.read_parquet(f"{data_dir}/{user_split}_split/train_agg.parquet", columns = user_cols)
    train_user_edges = pd.read_parquet(f"{data_dir}/{user_split}_split/train.parquet", columns = edge_cols)
    if user_split == "train_test_valid":
        test_user_df = pd.read_parquet(f"{data_dir}/{user_split}_split/valid_agg.parquet", columns = user_cols)
        test_user_edges = pd.read_parquet(f"{data_dir}/{user_split}_split/valid.parquet", columns = edge_cols)
    else:
        test_user_df = pd.read_parquet(f"{data_dir}/{user_split}_split/test_agg.parquet", columns = user_cols)
        test_user_edges = pd.read_parquet(f"{data_dir}/{user_split}_split/test.parquet", columns = edge_cols)
    
    # Get embedding
    prod_embed_name = f"{prod_embed}_features_{embed_tye}"
    user_embed_name = f"user_reviews_features_{embed_tye}"
    
    # make user df
    product_embedding = torch.load(f"{data_dir}/{embedding_dir}/product/{prod_embed_name}.pt")
    train_user_embedding = torch.load(f"{data_dir}/{embedding_dir}/{user_split}_split/train_{user_embed_name}.pt")
    if user_split == "train_test_valid":
        test_user_embedding = torch.load(f"{data_dir}/{embedding_dir}/{user_split}_split/valid_{user_embed_name}.pt")
    else:
        test_user_embedding = torch.load(f"{data_dir}/{embedding_dir}/{user_split}_split/test_{user_embed_name}.pt")

    train_user_df["embedding"] = list(train_user_embedding.numpy())
    test_user_df["embedding"] = list(test_user_embedding.numpy())

    user_df = create_user_df(train_user_df, test_user_df)
    
    # Set up id mapping
    offset = user_df.user_id.nunique()
    user_id_to_idx = {unique_id : idx for idx, unique_id in enumerate(user_df.user_id.unique())}
    prod_id_to_idx = {unique_id : offset + idx for idx, unique_id in enumerate(product_df.parent_asin.unique())}
    
    # Add to df
    product_df["prod_idx"] = product_df.parent_asin.apply(lambda x: prod_id_to_idx[x])
    train_user_edges["user_idx"] = train_user_edges.user_id.apply(lambda x: user_id_to_idx[x])
    test_user_edges["user_idx"] = test_user_edges.user_id.apply(lambda x: user_id_to_idx[x])
    train_user_edges["prod_idx"] = train_user_edges.parent_asin.apply(lambda x: prod_id_to_idx[x])
    test_user_edges["prod_idx"] = test_user_edges.parent_asin.apply(lambda x: prod_id_to_idx[x])
    
    # Concat product nodes 
    product_nodes = torch.cat([torch.tensor(product_df.drop(["parent_asin", "prod_idx"], axis = 1).to_numpy()), product_embedding], dim = 1)
    
    # concat user nodes 
    user_embed = torch.tensor(np.vstack(user_df["embedding"].values))
    user_info = torch.tensor(user_df.drop(["user_id", "embedding"], axis = 1).to_numpy())
    user_nodes = torch.cat([user_info, user_embed], dim = 1)
    
    # Create edge list
    train_edge_index, train_edge_weights = create_edge_lists(train_user_edges)
    test_edge_index, test_edge_weights = create_edge_lists(train_user_edges)
    
    # Move to gpu
    product_nodes = product_nodes.type(torch.float).to(device)
    user_nodes = user_nodes.type(torch.float).to(device)
    train_edge_index = train_edge_index.to(device)
    train_edge_weights = train_edge_weights.to(device)
    test_edge_index = test_edge_index.to(device)
    test_edge_weights = test_edge_weights.to(device)

    # model features
    num_users = len(user_df)
    num_products = len(product_df)
    user_feature_dim = user_nodes.shape[1]
    product_feature_dim = product_nodes.shape[1]
    embedding_dim = 64

    # Instantiate the model
    model = Base_GNN_Model(num_users, num_products, user_feature_dim, product_feature_dim, embedding_dim)

    # move the model 
    model.to(device)

    # Train model 
    train_loss, test_loss, best_model, best_epoch = train_model(model, train_edge_index, train_edge_weights, test_edge_index, test_edge_weights,
                                                                user_nodes, product_nodes, num_epochs = 1000, print_progress=False, give_epoch=True)
    final_test_loss = final_evaluation(model, test_edge_index, test_edge_weights, user_nodes, product_nodes, device, plot=False, print_test=False)
    
    # best loss 
    model.load_state_dict(best_model)
    best_test_loss = final_evaluation(model, test_edge_index, test_edge_weights, user_nodes, product_nodes, device, plot=False, print_test=False)

    return train_loss, test_loss, final_test_loss, best_epoch, best_test_loss