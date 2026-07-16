# src/data_pipeline/preprocess.py
import numpy as np

def align_vix_to_prices(vix_raw: np.ndarray, prices_length: int) -> np.ndarray:
    if len(vix_raw) > prices_length:
        vix_aligned = vix_raw[-prices_length:]
    elif len(vix_raw) < prices_length:
        vix_aligned = np.pad(vix_raw, (0, prices_length - len(vix_raw)), mode="edge")
    else:
        vix_aligned = vix_raw
    
    # reshaping just incase I dont download vix as a list giving (T,) instead of (T, N)
    return vix_aligned.reshape(-1, 1)

def scale_vix(vix: np.ndarray) -> np.ndarray:
    """
    The paper divided the vix by 100 to normalize it. 
    """
    return vix / 100.0

def prep_condition_vector(vix_raw: np.ndarray, prices_length: int) -> np.ndarray:
    """
    call this to get the nice and clean condition list.
    """
    aligned_matrix = align_vix_to_prices(vix_raw, prices_length)
    scaled_matrix = scale_vix(aligned_matrix)
    return scaled_matrix

