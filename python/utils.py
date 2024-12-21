import numpy as np

def one_hot_encode(data, categories):
    n_samples = len(data)
    n_categories = len(categories)
    encoded = np.zeros((n_samples, n_categories), dtype=int)
    category_to_index = {category: i for i, category in enumerate(categories)}
    
    for i, value in enumerate(data):
        if value in category_to_index:  # Ensure value is valid
            encoded[i, category_to_index[value]] = 1
        else:
            raise ValueError(f"Unknown category '{value}' found in data")
    return encoded
