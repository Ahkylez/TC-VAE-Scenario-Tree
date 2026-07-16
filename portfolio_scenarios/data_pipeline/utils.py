import numpy as np
from typing import Tuple

def data_split(data: np.ndarray, split_val: int | float) -> Tuple[np.ndarray, np.ndarray]:
    """
    We have is so it splits based on either an index int or a percentage float. 
    """
    n_samples = len(data)
    
    if isinstance(split_val, float):
        if not (0.0 < split_val < 1.0):
            raise ValueError("Percentage split must be between 0.0 and 1.0")
        split_idx = int(n_samples * split_val)
    else:
        split_idx = split_val
        
    train_data = data[:split_idx]
    test_data = data[split_idx:]
    
    return train_data, test_data