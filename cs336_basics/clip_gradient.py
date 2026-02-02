import torch

def gradient_clipping(parameters, max_l2_norm) -> None:
    """
    Clips the gradients of the given parameters to have a maximum L2 norm of `max_l2_norm`.
    
    Args:
        parameters (Iterable[torch.nn.Parameter]): An iterable of model parameters.
        max_l2_norm (float): The maximum allowed L2 norm for the gradients.
    """
    total_norm = 0.0
    grads = [p.grad for p in parameters if p.grad is not None]
    for grad in grads:
        total_norm += torch.sum(grad ** 2)
    total_norm = torch.sqrt(total_norm)        
    clip_coef = min(1, max_l2_norm / (total_norm + 1e-6))
    if clip_coef < 1:
        for grad in grads:
            grad.mul_(clip_coef)