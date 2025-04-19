import copy
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import seaborn as sns

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
                num_epochs=100, lr=0.01, optimiser = None, device = None, print_progress = False, print_freq = 10):
    
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

def final_evaluation(model, test_edge_index, test_edge_weights, user_features, product_features, device = None, plot = False):
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
        print(f"Test loss: {test_loss:.4f}")

    if plot:
        plot_actual_vs_predicted_ratings(test_predictions.cpu(), test_edge_weights.cpu())

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