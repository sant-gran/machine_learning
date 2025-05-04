import torch
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error
from math import sqrt

# RMSE for CF
def rmse(predictions, targets):
    """
    Compute Root Mean Squared Error (RMSE) between predicted and true values.

    Args:
        predictions (array-like or tensor): Predicted values
        targets (array-like or tensor): Ground truth values

    Returns:
        float: RMSE value
    """
    # Convert to numpy if torch tensor
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()

    return sqrt(mean_squared_error(targets, predictions))



