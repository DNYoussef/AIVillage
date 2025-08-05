# rag_system/error_handling/utils.py


import numpy as np


def compute_bonferroni_correction(overall_error_rate: float, num_steps: int) -> list[float]:
    """Compute Bonferroni correction for multiple steps.

    :param overall_error_rate: The overall target error rate.
    :param num_steps: The number of steps in the process.
    :return: A list of corrected error rates for each step.
    """
    return [overall_error_rate / num_steps] * num_steps


def normalize_error_rates(error_rates: list[float], target_sum: float) -> list[float]:
    """Normalize a list of error rates to sum up to a target value.

    :param error_rates: The list of error rates to normalize.
    :param target_sum: The target sum for the normalized error rates.
    :return: A list of normalized error rates.
    """
    total = sum(error_rates)
    return [rate * target_sum / total for rate in error_rates]


def compute_confidence_interval(observed_error: float, sample_size: int, confidence_level: float) -> tuple:
    """Compute the confidence interval for an observed error rate.

    :param observed_error: The observed error rate.
    :param sample_size: The size of the sample.
    :param confidence_level: The desired confidence level (e.g., 0.95 for 95% confidence).
    :return: A tuple containing the lower and upper bounds of the confidence interval.
    """
    z_score = np.abs(np.percentile(np.random.standard_normal(10000), (1 - confidence_level) / 2 * 100))
    margin_of_error = z_score * np.sqrt((observed_error * (1 - observed_error)) / sample_size)

    lower_bound = max(0, observed_error - margin_of_error)
    upper_bound = min(1, observed_error + margin_of_error)

    return (lower_bound, upper_bound)
