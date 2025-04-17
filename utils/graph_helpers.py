from torch.utils.data import DataLoader, TensorDataset

import torch
import torch.nn as nn

import matplotlib.pyplot as plt

import copy

def plot_loss(train_loss, validation_loss):
    # plot loss curves
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(train_loss)), train_loss, label='Training Loss')
    plt.plot(range(len(validation_loss)), validation_loss, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

# def train_model_with_batches(model, train_edge_index, train_edge_weights, val_edge_index, val_edge_weights, user_features, product_features, 
#                 num_epochs=100, lr=0.01, batch_size = 64, optimiser = None, device = None, print_progress = False):

#     """
#     to do 
#     i think need to shift the tensors back to GPU after getting the batch 
#     seems a bit too slow even though i see actitivy in the gpu
#     """
    
#     if device is None:
#         device = "cuda" if torch.cuda.is_available() else "cpu"
    
#     if optimiser is None:
#         optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    
#     # lists to store loss values for plotting
#     train_losses = []
#     valid_losses = []

#     # Hold for the best config 
#     best_model_state = model.state_dict()
#     best_valid_loss = float("inf")
#     best_model_epoch = None

#     # Move to the device 
#     model.to(device)
#     user_features = user_features.to(device)
#     product_features = product_features.to(device)
#     train_edge_index = train_edge_index.to(device)
#     train_edge_weights = train_edge_weights.to(device)
#     val_edge_index = val_edge_index.to(device)
#     val_edge_weights = val_edge_weights.to(device)

#     # Create datasets for batching 
#     train_dataset = TensorDataset(train_edge_index.T, train_edge_weights)
#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#     valid_dataset = TensorDataset(val_edge_index.T, val_edge_weights)
#     valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)

#     # training loop across epochs
#     for epoch in range(1, num_epochs + 1):
#         # training
#         model.train()

#         # Loss montioring
#         epoch_train_loss = 0
#         epoch_valid_loss = 0

#         for train_batch in train_loader:
#             # reset 
#             optimiser.zero_grad()

#             # batch
#             train_edge_batch, train_weight_batch = train_batch
#             train_edge_batch = train_edge_batch.T

#             # forward pass
#             train_batch_predictions = model(train_edge_batch, user_features, product_features)

#             # MSE 
#             train_batch_loss = nn.functional.mse_loss(train_batch_predictions, train_weight_batch)

#             # backward pass and optimisation
#             train_batch_loss.backward()
#             optimiser.step()
            
#             # batch loss
#             epoch_train_loss += train_batch_loss.item()
        
#         # training complete move to validation 
#         model.eval()
#         with torch.no_grad():
#             for valid_batch in valid_loader:
#                 # batch
#                 valid_edge_batch, valid_weight_batch = valid_batch
#                 valid_edge_batch = valid_edge_batch.T

#                 valid_batch_predictions = model(valid_edge_batch, user_features, product_features)
#                 valid_batch_loss = nn.functional.mse_loss(valid_batch_predictions, valid_weight_batch)
#                 epoch_valid_loss += valid_batch_loss.item()
        
#         # Append losses 
#         epoch_train_loss /= len(train_loader)
#         epoch_valid_loss /= len(valid_loader)
#         train_losses.append(epoch_train_loss)
#         valid_losses.append(epoch_valid_loss)

#         if epoch_valid_loss < best_valid_loss:
#             best_valid_loss = epoch_valid_loss
#             best_model_state = model.state_dict()
#             best_model_epoch = epoch

#         # print progress
#         if epoch % 10 == 0 and print_progress:
#             print(f'Epoch: {epoch}, Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_valid_loss:.4f}, best model epoch: {best_model_epoch}')
    
#     return train_losses, valid_losses, best_model_state

def train_model(model, train_edge_index, train_edge_weights, val_edge_index, val_edge_weights, user_features, product_features, 
                num_epochs=100, lr=0.01, optimiser = None, device = None, print_progress = False):
    
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if optimiser is None:
        optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    
    # lists to store loss values for plotting
    train_losses = []
    valid_losses = []

    # Hold for the best config 
    best_model_state = model.state_dict()
    best_valid_loss = float("inf")
    best_model_epoch = None

    # Move to the device 
    # model.to(device)
    # user_features = user_features.to(device)
    # product_features = product_features.to(device)
    # train_edge_index = train_edge_index.to(device)
    # train_edge_weights = train_edge_weights.to(device)
    # val_edge_index = val_edge_index.to(device)
    # val_edge_weights = val_edge_weights.to(device)

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
            valid_predictions = model(val_edge_index, user_features, product_features)
            valid_loss = nn.functional.mse_loss(valid_predictions, val_edge_weights)
    
        # Append losses 
        train_losses.append(train_loss.item())
        valid_losses.append(valid_loss.item())

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_model_state = copy.deepcopy(model.state_dict())
            best_model_epoch = epoch

        # print progress
        if epoch % 10 == 0 and print_progress:
            print(f'Epoch: {epoch}, Train Loss: {train_loss:.4f}, Val Loss: {valid_loss:.4f}, best model epoch: {best_model_epoch}')
    
    return train_losses, valid_losses, best_model_state

def final_evaluation(model, test_edge_index, test_edge_weights, user_features, product_features, best_state = None, device = None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # model.to(device)
    # test_edge_index = test_edge_index.to(device)
    # test_edge_weights = test_edge_weights.to(device)
    # user_features = user_features.to(device)
    # product_features =  product_features.to(device)

    # Final evaluation on test set
    model.eval()
    with torch.no_grad():
        test_predictions = model(test_edge_index, user_features, product_features)
        test_loss = nn.functional.mse_loss(test_predictions, test_edge_weights)
        print(f"Test loss: {test_loss:.4f}")

    if best_state is not None:
        temp_state = model.state_dict()
        model.load_state_dict(best_state)
        best_test_predictions = model(test_edge_index, user_features, product_features)
        best_test_loss = nn.functional.mse_loss(best_test_predictions, test_edge_weights)
        model.load_state_dict(temp_state)
        print(f"Best possible loss: {best_test_loss:.4f}")

    return test_loss

def plot_embedding_features(embedding_features):
    plt.figure(figsize=(8, 6))
    plt.scatter(embedding_features[:, 0], embedding_features[:, 1], alpha=0.5)
    plt.title("t-SNE Visualization of Embedding Features (2D)")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.show()
