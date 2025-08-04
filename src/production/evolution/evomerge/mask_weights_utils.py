import torch


def mask_input_with_mask_rate(
    input_tensor: torch.Tensor, mask_rate: float, use_rescale: bool, mask_strategy: str
) -> torch.Tensor:
    """Mask the input tensor based on the specified mask rate and strategy.

    :param input_tensor: The input tensor to be masked
    :param mask_rate: The rate of values to be masked (between 0 and 1)
    :param use_rescale: Whether to rescale the remaining values after masking
    :param mask_strategy: The strategy for masking ('random' or 'magnitude')
    :return: The masked tensor
    """
    if mask_strategy == "random":
        mask = torch.rand_like(input_tensor) > mask_rate
    elif mask_strategy == "magnitude":
        abs_tensor = torch.abs(input_tensor)
        threshold = torch.quantile(abs_tensor, mask_rate)
        mask = abs_tensor > threshold
    else:
        msg = f"Invalid mask strategy: {mask_strategy}"
        raise ValueError(msg)

    masked_tensor = input_tensor * mask

    if use_rescale:
        masked_tensor = masked_tensor / (1 - mask_rate)

    return masked_tensor
