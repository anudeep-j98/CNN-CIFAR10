import torch
import torch.nn.functional as F
import torch.optim as optim



def GetCorrectPredCount(pPrediction, pLabels):
    """Function to get the correct prediction count.

    Args:
        pPrediction (Object): Predicted tensors
        pLabels (Object): Actual labels of the images.

    Returns:
        Object(Tensor): If the predicted lables are actual labels then those items will be counted and returned.
    """
    return pPrediction.argmax(dim=1).eq(pLabels).sum().item()