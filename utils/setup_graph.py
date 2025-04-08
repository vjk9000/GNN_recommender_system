from sklearn.model_selection import train_test_split

def make_mapping(df, col):
    unique = df[col].unique()
    mapping = {unique_id : idx for idx, unique_id in enumerate(unique)}
    return mapping

def create_nodes(mapping, df, col):
    return [mapping[x] for x in df[col]]

def train_test_validation_split(edge_index, edge_weights, val_size = 0.3, test_size = 0.3, seed = 42):
    train_val_indices, test_indices = train_test_split(range(len(edge_index[0])), test_size = test_size, random_state = seed) # sep test
    train_indices, val_indices = train_test_split(train_val_indices, test_size = val_size, random_state = seed) # sep train val

    # train dataset
    train_edge_index = edge_index[:, train_indices]
    train_edge_weights = edge_weights[train_indices]

    # validation dataset
    val_edge_index = edge_index[:, val_indices]
    val_edge_weights = edge_weights[val_indices]

    # test dataset
    test_edge_index = edge_index[:, test_indices]
    test_edge_weights = edge_weights[test_indices]

    return train_edge_index, train_edge_weights, val_edge_index, val_edge_weights, test_edge_index, test_edge_weights