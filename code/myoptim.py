
"""Optimization settings: batch size, number of epochs, optimizer.

TODO: choose BATCH_SIZE and NUM_EPOCHS, and complete `get_optimizer`.
"""

import torch

# Training settings
BATCH_SIZE = 32
NUM_EPOCHS = 100


def get_optimizer(model):
    """Create the optimizer that updates the parameters of model.

    Args:
        model: the VAE defined in mymodel.py.

    Returns:
        a torch.optim.Optimizer.
    """
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001
    )

    return optimizer
