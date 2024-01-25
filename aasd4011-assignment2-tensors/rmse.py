import numpy as np

def rmse(predictions, targets):
    pred = np.array(predictions)
    tar = np.array(targets)
    n = len(pred)
    
    # Calculate the squared differences
    squared_diff = (pred - tar) ** 2
    
    # Calculate the mean squared difference
    mean_squared_diff = np.sum(squared_diff) / n
    
    # Calculate the square root of the mean squared difference
    rmse = np.sqrt(mean_squared_diff)
    
    return rmse